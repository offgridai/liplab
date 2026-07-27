#include "Lipsync/OffgridAILipsyncRuntimeAdapter.h"
#include "Lipsync/OffgridAILipsyncVersion.h"

#if OFFGRIDAI_WITH_NEURAL_LIPSYNC
#include "Lipsync/OffgridAINeuralStreamingAligner.h"
#endif

namespace
{
static int32 RegionContaining(
    const TArray<FOffgridAIStreamingSpeechRegion>& Regions,
    float TimeSec)
{
    for (int32 Index = 0; Index < Regions.Num(); ++Index)
    {
        const auto& Region = Regions[Index];
        const float EndSec = Region.bEnded
            ? Region.AudioBufferEndSec
            : FMath::Max(Region.AudioBufferEndSec, Region.AudioBufferLastSpeechSec);
        if (TimeSec >= Region.AudioBufferStartSec && TimeSec <= EndSec)
            return Index;
    }
    return INDEX_NONE;
}
}

FOffgridAILipsyncRuntimeSession::FOffgridAILipsyncRuntimeSession() = default;
FOffgridAILipsyncRuntimeSession::~FOffgridAILipsyncRuntimeSession() = default;

const TCHAR* FOffgridAILipsyncRuntimeSession::GetImplementationVersion()
{
    return OFFGRIDAI_LIPSYNC_IMPLEMENTATION_VERSION;
}

int32 FOffgridAILipsyncRuntimeSession::GetDiagnosticSchemaVersion()
{
    return OFFGRIDAI_LIPSYNC_DIAGNOSTIC_SCHEMA_VERSION;
}

void FOffgridAILipsyncRuntimeSession::Reset()
{
    NPCID = NAME_None;
    LineID = NAME_None;
    DialogueText.Reset();
    PrerollSec = 0.350f;
    PlaybackSec = 0.0f;
    bBegun = false;
    bCommittedTrackBuilt = false;
    bInputStreamClosed = false;
    Backend = EOffgridAILipsyncRuntimeBackend::Disabled;
    BackendError.Reset();
    TextPlan = FOffgridAITextVisemePlan();
    Detector.Reset();
    ResolvedSpeechRegions.Reset();
    CommittedTrack = FOffgridAICommittedVisemeTrack();
    RuntimeSpeechRegionDiagnosticRows.Reset();
    RuntimeBoundaryDiagnosticRows.Reset();
    RuntimeSyllableAssignmentDiagnosticRows.Reset();
    DiagnosticUpdateOrdinal = 0;
    StreamTailDiagnosticRow = FOffgridAIStreamTailDiagnosticRow();
    PCMChunkCount = 0;
    PCMBytesReceived = 0;
    PCMSamplesReceived = 0;
    LastPCMChunkSampleRate = 0;
    LastPCMChunkChannels = 0;
    LastPCMChunkStartSample = -1;
    LastPCMChunkEndSample = -1;
}

void FOffgridAILipsyncRuntimeSession::DisableNeuralRuntime(const FString& Reason)
{
    Backend = EOffgridAILipsyncRuntimeBackend::Disabled;
    BackendError = Reason;
    bCommittedTrackBuilt = false;
    ResolvedSpeechRegions.Reset();
    CommittedTrack = FOffgridAICommittedVisemeTrack();
    CommittedTrack.NPCID = NPCID;
    CommittedTrack.LineID = LineID;
}

void FOffgridAILipsyncRuntimeSession::BeginLine(
    const FOffgridAILipsyncRuntimeBeginInput& Input)
{
    Reset();
    NPCID = Input.NPCID;
    LineID = Input.LineID;
    DialogueText = Input.DialogueText;
    PrerollSec = FMath::Max(Input.PrerollSec, 0.0f);
    TextPlan = FOffgridAITextVisemePlanner::BuildPlan(FText::FromString(DialogueText));
    CommittedTrack.NPCID = NPCID;
    CommittedTrack.LineID = LineID;
    bBegun = true;

#if OFFGRIDAI_WITH_NEURAL_LIPSYNC
    if (!NeuralAligner)
        NeuralAligner = std::make_unique<FOffgridAINeuralStreamingAligner>();
    if (NeuralAligner->Begin(TextPlan, Input.NeuralCheckpointPath, BackendError))
        Backend = EOffgridAILipsyncRuntimeBackend::NeuralCuda;
    else
        DisableNeuralRuntime(BackendError);
#else
    DisableNeuralRuntime(FString(TEXT(
        "OffgridAI was built without the CUDA neural lipsync runtime")));
#endif
}

void FOffgridAILipsyncRuntimeSession::PushAudioPCM16(
    const TArray<uint8>& PCMChunk,
    int32 BytesToUse,
    int32 SampleRate,
    int32 NumChannels,
    int64 ChunkStartSample)
{
    if (!bBegun) return;

    ++PCMChunkCount;
    PCMBytesReceived += FMath::Max(BytesToUse, 0);
    LastPCMChunkSampleRate = SampleRate;
    LastPCMChunkChannels = NumChannels;
    const int32 Samples = NumChannels > 0
        ? BytesToUse / FMath::Max(NumChannels * static_cast<int32>(sizeof(int16)), 1)
        : 0;
    PCMSamplesReceived += Samples;
    LastPCMChunkStartSample = ChunkStartSample;
    LastPCMChunkEndSample = ChunkStartSample >= 0 ? ChunkStartSample + Samples : -1;

    // Disabled means exactly that: no analysis, no scheduler, and no track.
    if (Backend != EOffgridAILipsyncRuntimeBackend::NeuralCuda) return;

    // The detector is retained solely as the canonical causal PCM feature
    // extractor. Its occupancy and timing decisions are never consumed here.
    Detector.AppendPCM16(PCMChunk, BytesToUse, SampleRate, NumChannels, ChunkStartSample);
#if OFFGRIDAI_WITH_NEURAL_LIPSYNC
    if (!NeuralAligner || !NeuralAligner->ProcessAvailableFrames(
            Detector.GetFeatureFrames(), PlaybackSec, CommittedTrack, BackendError))
    {
        DisableNeuralRuntime(BackendError.IsEmpty()
            ? FString(TEXT("neural inference failed"))
            : BackendError);
        return;
    }
#endif
    RefreshResolvedSpeechRegions();
    bCommittedTrackBuilt = CommittedTrack.Events.Num() > 0;
}

void FOffgridAILipsyncRuntimeSession::CloseInputStream()
{
    bInputStreamClosed = true;
    if (Backend != EOffgridAILipsyncRuntimeBackend::NeuralCuda) return;
    Detector.Finalize(Detector.GetObservedAudioBufferEndSec());
#if OFFGRIDAI_WITH_NEURAL_LIPSYNC
    if (!NeuralAligner || !NeuralAligner->ProcessAvailableFrames(
            Detector.GetFeatureFrames(), PlaybackSec, CommittedTrack, BackendError))
    {
        DisableNeuralRuntime(BackendError.IsEmpty()
            ? FString(TEXT("neural close failed"))
            : BackendError);
        return;
    }
    // PCM end is known while the final audio is still queued ahead of the
    // playback cursor. Resolve the fixed-lag tail now so its last words are
    // committed before their render windows, rather than waiting until audio
    // playback has already ended. Finalize() remains idempotent and records
    // terminal diagnostics later at the true final playback time.
    NeuralAligner->Finalize(
        Detector.GetObservedAudioBufferEndSec(), PlaybackSec, CommittedTrack);
#endif
    RefreshResolvedSpeechRegions();
    bCommittedTrackBuilt = CommittedTrack.Events.Num() > 0;
}

void FOffgridAILipsyncRuntimeSession::Update(float CurrentPlaybackSec)
{
    if (!bBegun) return;
    PlaybackSec = FMath::Max(CurrentPlaybackSec, 0.0f);
    if (Backend == EOffgridAILipsyncRuntimeBackend::NeuralCuda)
    {
        RefreshResolvedSpeechRegions();
        bCommittedTrackBuilt = CommittedTrack.Events.Num() > 0;
    }
    RecordRuntimeDiagnostics(PlaybackSec, false);
}

void FOffgridAILipsyncRuntimeSession::Finalize(float FinalPlaybackSec)
{
    if (!bBegun) return;
    PlaybackSec = FMath::Max(FinalPlaybackSec, PlaybackSec);
    bInputStreamClosed = true;
    if (Backend != EOffgridAILipsyncRuntimeBackend::NeuralCuda)
    {
        RecordRuntimeDiagnostics(PlaybackSec, true);
        return;
    }

    Detector.Finalize(Detector.GetObservedAudioBufferEndSec());
#if OFFGRIDAI_WITH_NEURAL_LIPSYNC
    if (!NeuralAligner || !NeuralAligner->ProcessAvailableFrames(
            Detector.GetFeatureFrames(), PlaybackSec, CommittedTrack, BackendError))
    {
        DisableNeuralRuntime(BackendError.IsEmpty()
            ? FString(TEXT("neural final inference failed"))
            : BackendError);
        RecordRuntimeDiagnostics(PlaybackSec, true);
        return;
    }
    NeuralAligner->Finalize(
        Detector.GetObservedAudioBufferEndSec(), PlaybackSec, CommittedTrack);
#endif
    RefreshResolvedSpeechRegions();
    bCommittedTrackBuilt = CommittedTrack.Events.Num() > 0;
    RecordRuntimeDiagnostics(PlaybackSec, true);
}

void FOffgridAILipsyncRuntimeSession::RefreshResolvedSpeechRegions()
{
    ResolvedSpeechRegions.Reset();
#if OFFGRIDAI_WITH_NEURAL_LIPSYNC
    if (Backend == EOffgridAILipsyncRuntimeBackend::NeuralCuda && NeuralAligner)
        ResolvedSpeechRegions = NeuralAligner->GetSpeechRegions();
#endif
}

void FOffgridAILipsyncRuntimeSession::RecordRuntimeDiagnostics(
    float CurrentPlaybackSec,
    bool bFinalReplay)
{
    ++DiagnosticUpdateOrdinal;
    StreamTailDiagnosticRow.LineID = LineID;
    StreamTailDiagnosticRow.PCMChunkCount = PCMChunkCount;
    StreamTailDiagnosticRow.PCMBytesReceived = PCMBytesReceived;
    StreamTailDiagnosticRow.PCMSamplesReceived = PCMSamplesReceived;
    StreamTailDiagnosticRow.LastSampleRate = LastPCMChunkSampleRate;
    StreamTailDiagnosticRow.LastNumChannels = LastPCMChunkChannels;
    StreamTailDiagnosticRow.LastChunkStartSample = LastPCMChunkStartSample;
    StreamTailDiagnosticRow.LastChunkEndSample = LastPCMChunkEndSample;
    StreamTailDiagnosticRow.ObservedAudioBufferEndSec =
        Backend == EOffgridAILipsyncRuntimeBackend::NeuralCuda
            ? Detector.GetObservedAudioBufferEndSec()
            : 0.0f;
    StreamTailDiagnosticRow.FirstSpeechAudioBufferStartSec =
        ResolvedSpeechRegions.Num() > 0
            ? ResolvedSpeechRegions[0].AudioBufferStartSec
            : 0.0f;
    StreamTailDiagnosticRow.SpeechRegionCount = ResolvedSpeechRegions.Num();
    StreamTailDiagnosticRow.bInputStreamClosed = bInputStreamClosed;
    StreamTailDiagnosticRow.DiagnosticKind = FName(TEXT("neural_stream_tail"));

    for (const auto& Region : ResolvedSpeechRegions)
    {
        FOffgridAIRuntimeSpeechRegionDiagnosticRow Row;
        Row.LineID = LineID;
        Row.UpdateOrdinal = DiagnosticUpdateOrdinal;
        Row.bFinalReplay = bFinalReplay;
        Row.CurrentPlaybackSec = CurrentPlaybackSec;
        Row.RegionIndex = Region.SpeechRegionIndex;
        Row.RegionOpenSec = Region.AudioBufferStartSec;
        Row.RegionCloseSec = Region.AudioBufferEndSec;
        Row.LastSpeechSec = Region.AudioBufferLastSpeechSec;
        Row.ProvisionalEndSec = Region.ProvisionalEndSec;
        Row.EndDecisionSec = Region.EndDecisionSec;
        Row.ReopenCount = Region.ReopenCount;
        Row.bStarted = Region.bStarted;
        Row.bEnded = Region.bEnded;
        Row.bContainsPlaybackSec = RegionContaining(
            ResolvedSpeechRegions, CurrentPlaybackSec) == Region.SpeechRegionIndex;
        Row.CommittedEventCount = CommittedTrack.Events.Num();
        Row.CloseReason = Region.EndReason;
        Row.DiagnosticKind = FName(TEXT("neural_speech_region"));
        RuntimeSpeechRegionDiagnosticRows.Add(Row);
    }

    FOffgridAIRuntimeBoundaryDiagnosticRow StateRow;
    StateRow.LineID = LineID;
    StateRow.UpdateOrdinal = DiagnosticUpdateOrdinal;
    StateRow.bFinalReplay = bFinalReplay;
    StateRow.CurrentPlaybackSec = CurrentPlaybackSec;
    StateRow.bPlayheadStarted = bCommittedTrackBuilt;
    StateRow.ActiveSpeechRegionIndex = RegionContaining(
        ResolvedSpeechRegions, CurrentPlaybackSec);
    if (ResolvedSpeechRegions.IsValidIndex(StateRow.ActiveSpeechRegionIndex))
    {
        const auto& Region = ResolvedSpeechRegions[StateRow.ActiveSpeechRegionIndex];
        StateRow.bAudioSpeechActive = true;
        StateRow.ActiveRegionStartSec = Region.AudioBufferStartSec;
        StateRow.ActiveRegionEndSec = Region.AudioBufferEndSec;
    }
    if (CommittedTrack.Events.Num() > 0)
    {
        const auto& Last = CommittedTrack.Events.Last();
        StateRow.SchedulerNextEventIndex = Last.EventIndex + 1;
        StateRow.SchedulerNextPhoneIndex = Last.SourcePhoneIndex + 1;
        StateRow.SchedulerCommitLeadSec = Last.CommitLeadSeconds;
    }
    StateRow.SchedulerCommitFrontierSec = CurrentPlaybackSec + PrerollSec;
    StateRow.SchedulerBlockReason = Backend == EOffgridAILipsyncRuntimeBackend::NeuralCuda
        ? FString(TEXT("neural_cuda"))
        : BackendError;
    StateRow.CommittedEventCount = CommittedTrack.Events.Num();
    StateRow.DiagnosticKind = FName(TEXT("neural_primary_runtime"));
    RuntimeBoundaryDiagnosticRows.Add(StateRow);
}
