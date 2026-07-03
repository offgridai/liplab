#include "Lipsync/OffgridAILipsyncRuntimeAdapter.h"

namespace
{
static const FName OccupancyReason(TEXT("speech_occupancy_playhead"));

static float SpanForPose(const FName& PoseID)
{
    const FString P = PoseID.ToString();
    if (P.Contains(TEXT("22_MBP"))) return 0.110f;
    if (P.Contains(TEXT("20_FV"))) return 0.105f;
    if (P.Contains(TEXT("12_Ww")) || P.Contains(TEXT("11_Oo")) || P.Contains(TEXT("09_Oh"))) return 0.125f;
    return 0.135f;
}

static float LeadForPose(const FName& PoseID)
{
    const FString P = PoseID.ToString();
    if (P.Contains(TEXT("22_MBP"))) return 0.030f;
    if (P.Contains(TEXT("20_FV"))) return 0.020f;
    if (P.Contains(TEXT("12_Ww"))) return 0.018f;
    return 0.0f;
}

static float FallbackEventWeight(const FOffgridAITextVisemeEvent& Event)
{
    const FString Pose = Event.PoseID.ToString();
    if (Pose.Contains(TEXT("22_MBP"))) return 0.080f;
    if (Pose.Contains(TEXT("20_FV")) || Pose.Contains(TEXT("14_ChJjSh"))) return 0.085f;
    if (Pose.Contains(TEXT("12_Ww")) || Pose.Contains(TEXT("11_Oo")) || Pose.Contains(TEXT("09_Oh"))) return 0.105f;
    if (Pose.Contains(TEXT("03_Ee")) || Pose.Contains(TEXT("05_Ay"))) return 0.095f;
    return 0.100f;
}

static void BuildEventProgressNorms(const FOffgridAITextVisemePlan& Plan, TArray<float>& OutCenterNorms, float& OutTotalWeight)
{
    const int32 EventCount = Plan.Events.Num();
    OutCenterNorms.Init(0.0f, EventCount);
    OutTotalWeight = 0.0f;
    if (EventCount <= 0)
    {
        return;
    }

    TArray<int32> WordPhoneBegin;
    TArray<int32> WordPhoneEnd;
    WordPhoneBegin.Init(INDEX_NONE, Plan.WordSyllableCounts.Num());
    WordPhoneEnd.Init(INDEX_NONE, Plan.WordSyllableCounts.Num());
    for (int32 PhoneIndex = 0; PhoneIndex < Plan.ExpectedPhones.Num(); ++PhoneIndex)
    {
        const FOffgridAIExpectedPhone& Phone = Plan.ExpectedPhones[PhoneIndex];
        if (!WordPhoneBegin.IsValidIndex(Phone.WordIndex))
        {
            continue;
        }
        int32& Begin = WordPhoneBegin[Phone.WordIndex];
        int32& End = WordPhoneEnd[Phone.WordIndex];
        if (Begin == INDEX_NONE) Begin = PhoneIndex;
        End = PhoneIndex + 1;
    }

    TArray<float> EventWeights;
    EventWeights.Init(0.0f, EventCount);
    for (int32 EventIndex = 0; EventIndex < EventCount; ++EventIndex)
    {
        const FOffgridAITextVisemeEvent& Event = Plan.Events[EventIndex];
        const int32 WordIndex = Event.WordIndex;
        int32 RangeStart = Event.SourcePhoneGlobalIndex;
        int32 RangeEnd = INDEX_NONE;

        for (int32 NextIndex = EventIndex + 1; NextIndex < EventCount; ++NextIndex)
        {
            const FOffgridAITextVisemeEvent& NextEvent = Plan.Events[NextIndex];
            if (NextEvent.WordIndex != WordIndex)
            {
                continue;
            }
            if (NextEvent.SourcePhoneGlobalIndex != INDEX_NONE)
            {
                RangeEnd = NextEvent.SourcePhoneGlobalIndex;
                break;
            }
        }

        if (RangeStart != INDEX_NONE && RangeEnd == INDEX_NONE && WordPhoneEnd.IsValidIndex(WordIndex))
        {
            RangeEnd = WordPhoneEnd[WordIndex];
        }

        float Weight = 0.0f;
        if (RangeStart != INDEX_NONE && RangeEnd != INDEX_NONE && RangeEnd > RangeStart)
        {
            for (int32 PhoneIndex = RangeStart; PhoneIndex < RangeEnd && Plan.ExpectedPhones.IsValidIndex(PhoneIndex); ++PhoneIndex)
            {
                Weight += FMath::Max(Plan.ExpectedPhones[PhoneIndex].WeightSeconds, 0.020f);
            }
        }

        if (Weight <= KINDA_SMALL_NUMBER)
        {
            const float FallbackWord = Plan.WordSyllableCounts.IsValidIndex(WordIndex)
                ? 0.075f * FMath::Max(Plan.WordSyllableCounts[WordIndex], 1)
                : 0.0f;
            Weight = FMath::Max(FallbackEventWeight(Event), FallbackWord);
        }

        EventWeights[EventIndex] = Weight;
        OutTotalWeight += Weight;
    }

    OutTotalWeight = FMath::Max(OutTotalWeight, 0.001f);
    float CumulativeWeight = 0.0f;
    for (int32 EventIndex = 0; EventIndex < EventCount; ++EventIndex)
    {
        const float Weight = EventWeights[EventIndex];
        const float CenterWeight = CumulativeWeight + Weight * 0.5f;
        OutCenterNorms[EventIndex] = FMath::Clamp(CenterWeight / OutTotalWeight, 0.0f, 1.0f);
        CumulativeWeight += Weight;
    }
}

struct FEffectiveSpeechRegion
{
    float StartSec = 0.0f;
    float EndSec = 0.0f;
};

static float IslandObservedEnd(const FOffgridAIStreamingSpeechIsland& Island, float ObservedEndSec, bool bFinal)
{
    if (Island.bEnded || bFinal)
    {
        return FMath::Clamp(Island.AudioBufferEndSec, Island.AudioBufferStartSec, ObservedEndSec);
    }
    return FMath::Max(Island.AudioBufferStartSec, ObservedEndSec);
}

static void BuildEffectiveSpeechRegions(const TArray<FOffgridAIStreamingSpeechIsland>* Islands, float ObservedEndSec, bool bFinal, TArray<FEffectiveSpeechRegion>& OutRegions)
{
    OutRegions.Reset();
    if (!Islands) return;

    // Literal speech occupancy: each detector island is a hard region.
    // Do not merge low-confidence/brief gaps and do not pad edges; even short
    // pauses should stop active-speech progress and resume from the next viseme.
    for (const FOffgridAIStreamingSpeechIsland& Island : *Islands)
    {
        if (!Island.bStarted) continue;

        const float S = FMath::Clamp(Island.AudioBufferStartSec, 0.0f, ObservedEndSec);
        const float E = FMath::Clamp(IslandObservedEnd(Island, ObservedEndSec, bFinal), S, FMath::Max(ObservedEndSec, S));
        if (E - S <= 0.010f) continue;

        FEffectiveSpeechRegion R;
        R.StartSec = S;
        R.EndSec = E;
        OutRegions.Add(R);
    }
}

static float ComputeObservedActiveSpeechSeconds(const TArray<FEffectiveSpeechRegion>& Regions)
{
    float Active = 0.0f;
    for (const FEffectiveSpeechRegion& R : Regions)
    {
        Active += FMath::Max(R.EndSec - R.StartSec, 0.0f);
    }
    return Active;
}

static float ComputeFirstSpeechStart(const TArray<FEffectiveSpeechRegion>& Regions)
{
    return Regions.Num() > 0 ? Regions[0].StartSec : -1.0f;
}

static float ComputeLastSpeechEnd(const TArray<FEffectiveSpeechRegion>& Regions, float ObservedEndSec)
{
    return Regions.Num() > 0 ? Regions.Last().EndSec : ObservedEndSec;
}

static bool MapActiveSpeechTimeToObservedClock(const TArray<FEffectiveSpeechRegion>& Regions, float TargetActiveSec, float& OutClockSec)
{
    float Remaining = FMath::Max(TargetActiveSec, 0.0f);
    for (const FEffectiveSpeechRegion& R : Regions)
    {
        const float Dur = FMath::Max(R.EndSec - R.StartSec, 0.0f);
        if (Dur <= KINDA_SMALL_NUMBER) continue;

        if (Remaining <= Dur)
        {
            OutClockSec = R.StartSec + Remaining;
            return true;
        }
        Remaining -= Dur;
    }
    return false;
}

static FName SourcePhoneClassName(const FString& BasePhone)
{
    if (BasePhone == TEXT("B") || BasePhone == TEXT("M") || BasePhone == TEXT("P")) return FName(TEXT("bilabial"));
    if (BasePhone == TEXT("F") || BasePhone == TEXT("V")) return FName(TEXT("labiodental"));
    if (BasePhone == TEXT("S") || BasePhone == TEXT("Z") || BasePhone == TEXT("SH") || BasePhone == TEXT("ZH")) return FName(TEXT("sibilant"));
    if (BasePhone == TEXT("W") || BasePhone == TEXT("R") || BasePhone == TEXT("L") || BasePhone == TEXT("Y")) return FName(TEXT("glide_liquid"));
    if (BasePhone.Len() > 0) return FName(TEXT("vowel_or_other"));
    return NAME_None;
}

static void FillEventFromText(
    const FOffgridAITextVisemePlan& Plan,
    int32 EventIndex,
    const FOffgridAITextVisemeEvent& T,
    float OrderNorm,
    float Center,
    float Span,
    float ObservedActiveSec,
    float RequiredActiveSec,
    float TotalPlannedActiveSec,
    const FOffgridAILipsyncRuntimeUpdateInput& Input,
    FOffgridAIAlignedVisemeEvent& Out)
{
    Out.EventIndex = EventIndex;
    Out.PoseID = T.PoseID;
    Out.Strength = T.Strength;
    Out.SourceWord = T.SourceText;
    Out.WordIndex = T.WordIndex;
    Out.PhraseIndex = T.PhraseIndex;
    Out.SentenceIndex = T.SentenceIslandIndex;
    Out.bIsStrongVisibleEvent = T.bIsStrongVisibleEvent;

    Out.TextCenterNorm = OrderNorm;
    Out.TextDiagnosticCenterSeconds = OrderNorm * FMath::Max(Plan.EstimatedDurationSeconds, 0.001f);
    Out.FinalRenderCenterSeconds = Center;
    Out.RenderStartSeconds = FMath::Max(Center - Span * 0.50f, 0.0f);
    Out.RenderEndSeconds = Center + Span * 0.50f;

    Out.SourcePhoneIndex = T.SourcePhoneGlobalIndex;
    Out.SourcePhoneBase = T.SourcePhoneBase;
    Out.SourcePhoneClass = SourcePhoneClassName(T.SourcePhoneBase);

    Out.bMappedToObservedSpeech = true;

    Out.CommitPlaybackSeconds = Input.CurrentPlaybackSec;
    Out.CommitLeadSeconds = Center - Input.CurrentPlaybackSec;
    Out.CommitReason = OccupancyReason;
    Out.RequiredActiveElapsedSeconds = RequiredActiveSec;
    Out.ObservedActiveElapsedSeconds = ObservedActiveSec;
    Out.ActiveProgressDeficitSeconds = FMath::Max(RequiredActiveSec - ObservedActiveSec, 0.0f);
    Out.RequiredProgressNorm = TotalPlannedActiveSec > KINDA_SMALL_NUMBER ? RequiredActiveSec / TotalPlannedActiveSec : 1.0f;
    Out.ObservedProgressNorm = TotalPlannedActiveSec > KINDA_SMALL_NUMBER ? ObservedActiveSec / TotalPlannedActiveSec : 1.0f;
    Out.ActiveProgressRatio = RequiredActiveSec > KINDA_SMALL_NUMBER ? ObservedActiveSec / RequiredActiveSec : 1.0f;
}
}

void FOffgridAILipsyncRuntimeSession::Reset()
{
    NPCID = NAME_None;
    LineID = NAME_None;
    DialogueText.Reset();
    PrerollSec = 0.350f;
    PlaybackSec = 0.0f;
    bBegun = false;
    bPlaybackStarted = false;
    bCommittedTrackBuilt = false;
    bInputStreamClosed = false;

    TextPlan = FOffgridAITextVisemePlan();
    Detector.Reset();
    ResolvedSpeechIslands.Reset();
    CommittedTrack = FOffgridAIAlignedVisemeTrack();
    AudioOccupancyDiagnosticRows.Reset();
    AudioOccupancyDiagnosticUpdateOrdinal = 0;
    StreamTailDiagnosticRow = FOffgridAIStreamTailDiagnosticRow();

    PCMChunkCount = 0;
    PCMBytesReceived = 0;
    PCMSamplesReceived = 0;
    LastPCMChunkSampleRate = 0;
    LastPCMChunkChannels = 0;
    LastPCMChunkStartSample = -1;
    LastPCMChunkEndSample = -1;
}

void FOffgridAILipsyncRuntimeSession::BeginLine(const FOffgridAILipsyncRuntimeBeginInput& Input)
{
    Reset();
    NPCID = Input.NPCID;
    LineID = Input.LineID;
    DialogueText = Input.DialogueText;
    PrerollSec = FMath::Max(Input.PrerollSec, 0.0f);
    TextPlan = FOffgridAITextVisemePlanner::BuildPlan(FText::FromString(DialogueText));
    RefreshResolvedSpeechIslands();
    CommittedTrack.NPCID = NPCID;
    CommittedTrack.LineID = LineID;
    bBegun = true;
}

void FOffgridAILipsyncRuntimeSession::PushAudioPCM16(const TArray<uint8>& PCMChunk, int32 BytesToUse, int32 SampleRate, int32 NumChannels, int64 ChunkStartSample)
{
    if (!bBegun) return;

    Detector.AppendPCM16(PCMChunk, BytesToUse, SampleRate, NumChannels, ChunkStartSample);
    RefreshResolvedSpeechIslands();

    ++PCMChunkCount;
    PCMBytesReceived += FMath::Max(BytesToUse, 0);
    LastPCMChunkSampleRate = SampleRate;
    LastPCMChunkChannels = NumChannels;
    const int32 Frames = NumChannels > 0 ? BytesToUse / FMath::Max(NumChannels * static_cast<int32>(sizeof(int16)), 1) : 0;
    PCMSamplesReceived += Frames;
    LastPCMChunkStartSample = ChunkStartSample;
    LastPCMChunkEndSample = ChunkStartSample >= 0 ? ChunkStartSample + Frames : -1;
}

void FOffgridAILipsyncRuntimeSession::CloseInputStream()
{
    bInputStreamClosed = true;
    Detector.Finalize();
    RefreshResolvedSpeechIslands();
}

void FOffgridAILipsyncRuntimeSession::Update(float CurrentPlaybackSec)
{
    PlaybackSec = FMath::Max(CurrentPlaybackSec, 0.0f);
    UpdatePlaybackGate(Detector.GetObservedAudioBufferEndSec());
    RefreshResolvedSpeechIslands();

    FOffgridAILipsyncRuntimeUpdateInput Input;
    Input.TextPlan = &TextPlan;
    Input.SpeechIslands = &ResolvedSpeechIslands;
    Input.AudioFeatureFrames = &Detector.GetFeatureFrames();
    Input.CurrentPlaybackSec = PlaybackSec;
    Input.PrerollSec = PrerollSec;
    Input.ObservedAudioBufferEndSec = Detector.GetObservedAudioBufferEndSec();
    Input.bInputStreamClosed = bInputStreamClosed;
    Input.NPCID = NPCID;
    Input.LineID = LineID;

    FOffgridAILipsyncRuntimeAdapter::UpdateCommittedTrack(Input, CommittedTrack, bCommittedTrackBuilt);
    RecordRuntimeDiagnostics(PlaybackSec, false);
}

void FOffgridAILipsyncRuntimeSession::Finalize(float FinalPlaybackSec)
{
    PlaybackSec = FMath::Max(FinalPlaybackSec, PlaybackSec);
    bInputStreamClosed = true;
    Detector.Finalize(Detector.GetObservedAudioBufferEndSec());
    RefreshResolvedSpeechIslands();

    FOffgridAILipsyncRuntimeUpdateInput Input;
    Input.TextPlan = &TextPlan;
    Input.SpeechIslands = &ResolvedSpeechIslands;
    Input.AudioFeatureFrames = &Detector.GetFeatureFrames();
    Input.CurrentPlaybackSec = PlaybackSec;
    Input.PrerollSec = PrerollSec;
    Input.ObservedAudioBufferEndSec = Detector.GetObservedAudioBufferEndSec();
    Input.bInputStreamClosed = true;
    Input.bPlaybackFinalized = true;
    Input.NPCID = NPCID;
    Input.LineID = LineID;

    FOffgridAILipsyncRuntimeAdapter::UpdateCommittedTrack(Input, CommittedTrack, bCommittedTrackBuilt);
    RecordRuntimeDiagnostics(PlaybackSec, true);
}

void FOffgridAILipsyncRuntimeSession::UpdatePlaybackGate(float ObservedEndSec)
{
    if (bPlaybackStarted) return;
    const bool bHasSpeech = Detector.HasObservedFirstSpeechStart() || ResolvedSpeechIslands.Num() > 0;
    if (bHasSpeech || ObservedEndSec >= PrerollSec || bInputStreamClosed)
    {
        bPlaybackStarted = true;
    }
}

void FOffgridAILipsyncRuntimeSession::RefreshResolvedSpeechIslands()
{
    ResolvedSpeechIslands = Detector.GetIslands();
}

void FOffgridAILipsyncRuntimeSession::RecordRuntimeDiagnostics(float CurrentPlaybackSec, bool bFinalReplay)
{
    StreamTailDiagnosticRow.LineID = LineID;
    StreamTailDiagnosticRow.PCMChunkCount = PCMChunkCount;
    StreamTailDiagnosticRow.PCMBytesReceived = PCMBytesReceived;
    StreamTailDiagnosticRow.PCMSamplesReceived = PCMSamplesReceived;
    StreamTailDiagnosticRow.LastSampleRate = LastPCMChunkSampleRate;
    StreamTailDiagnosticRow.LastNumChannels = LastPCMChunkChannels;
    StreamTailDiagnosticRow.LastChunkStartSample = LastPCMChunkStartSample;
    StreamTailDiagnosticRow.LastChunkEndSample = LastPCMChunkEndSample;
    StreamTailDiagnosticRow.ObservedAudioBufferEndSec = Detector.GetObservedAudioBufferEndSec();
    StreamTailDiagnosticRow.FirstSpeechAudioBufferStartSec = Detector.GetFirstSpeechAudioBufferStartSec();
    StreamTailDiagnosticRow.SpeechIslandCount = Detector.GetIslands().Num();
    StreamTailDiagnosticRow.bInputStreamClosed = bInputStreamClosed;
    StreamTailDiagnosticRow.DiagnosticKind = FName(TEXT("runtime_stream_tail"));

    AudioOccupancyDiagnosticRows.Reset();
    for (const FOffgridAIAlignedVisemeEvent& E : CommittedTrack.Events)
    {
        FOffgridAIAudioOccupancyDiagnosticRow R;
        R.LineID = LineID;
        R.UpdateOrdinal = AudioOccupancyDiagnosticUpdateOrdinal;
        R.bFinalReplay = bFinalReplay;
        R.CurrentPlaybackSec = CurrentPlaybackSec;
        R.PrerollSec = PrerollSec;
        R.SourceEventIndex = E.EventIndex;
        R.Word = E.SourceWord;
        R.PoseID = E.PoseID;
        R.PlannedCenterSec = E.TextDiagnosticCenterSeconds;
        R.CommittedCenterSec = E.FinalRenderCenterSeconds;
        R.RenderStartSec = E.RenderStartSeconds;
        R.RenderEndSec = E.RenderEndSeconds;
        R.CommitReason = E.CommitReason;
        R.PlaybackMode = E.CommitReason;
        R.AudioActiveSec = E.ObservedActiveElapsedSeconds;
        R.TextPlayheadSec = static_cast<float>(E.EventIndex + 1);
        R.RequiredActiveElapsedSec = E.RequiredActiveElapsedSeconds;
        R.ObservedActiveElapsedSec = E.ObservedActiveElapsedSeconds;
        R.ActiveProgressDeficitSec = E.ActiveProgressDeficitSeconds;
        R.RequiredProgressNorm = E.RequiredProgressNorm;
        R.ObservedProgressNorm = E.ObservedProgressNorm;
        R.ActiveProgressRatio = E.ActiveProgressRatio;
        R.bMappedToObservedSpeech = E.bMappedToObservedSpeech;
        R.DiagnosticKind = E.CommitReason;
        AudioOccupancyDiagnosticRows.Add(R);
    }
    ++AudioOccupancyDiagnosticUpdateOrdinal;
}

void FOffgridAILipsyncRuntimeAdapter::UpdateCommittedTrack(const FOffgridAILipsyncRuntimeUpdateInput& Input, FOffgridAIAlignedVisemeTrack& InOutTrack, bool& bInOutTrackBuilt)
{
    if (!Input.TextPlan) return;

    const FOffgridAITextVisemePlan& Plan = *Input.TextPlan;
    InOutTrack.NPCID = Input.NPCID;
    InOutTrack.LineID = Input.LineID;

    const int32 EventCount = Plan.Events.Num();
    if (EventCount == 0)
    {
        bInOutTrackBuilt = true;
        return;
    }

    TArray<float> EventCenterNorms;
    float EventTotalWeight = 0.0f;
    BuildEventProgressNorms(Plan, EventCenterNorms, EventTotalWeight);

    const bool bFinal = Input.bInputStreamClosed || Input.bPlaybackFinalized;
    const float ObservedEnd = FMath::Max(Input.ObservedAudioBufferEndSec, 0.0f);

    TArray<FEffectiveSpeechRegion> EffectiveRegions;
    BuildEffectiveSpeechRegions(Input.SpeechIslands, ObservedEnd, bFinal, EffectiveRegions);

    const float ObservedActiveSec = ComputeObservedActiveSpeechSeconds(EffectiveRegions);
    const float FirstSpeechStart = ComputeFirstSpeechStart(EffectiveRegions);
    const float LastSpeechEnd = ComputeLastSpeechEnd(EffectiveRegions, ObservedEnd);

    if (ObservedActiveSec <= 0.001f && !bFinal)
    {
        return;
    }

    InOutTrack.SpeechStartSeconds = FirstSpeechStart >= 0.0f ? FirstSpeechStart : 0.0f;
    InOutTrack.SpeechEndSeconds = LastSpeechEnd;

    // Live: pace by active speech occupancy. The estimated text duration is only
    // a conservative denominator until enough observed speech has arrived.
    // Final: distribute the full event list across the actual observed active speech.
    const float EstimatedActiveSec = FMath::Max(Plan.EstimatedDurationSeconds, 0.001f);
    const float TotalPlannedActiveSec = bFinal
        ? FMath::Max(ObservedActiveSec, 0.001f)
        : FMath::Max(EstimatedActiveSec, ObservedActiveSec);

    const float CommitLagSec = bFinal ? 0.0f : 0.030f;
    const float CommitSafeActiveSec = FMath::Max(ObservedActiveSec - CommitLagSec, 0.0f);
    const float MaxLiveLeadSec = bFinal ? 999.0f : FMath::Max(Input.PrerollSec + 0.120f, 0.250f);

    int32 NextEventIndex = InOutTrack.Events.Num();
    float LastCenter = InOutTrack.Events.Num() > 0 ? InOutTrack.Events.Last().FinalRenderCenterSeconds : -1.0f;

    while (Plan.Events.IsValidIndex(NextEventIndex))
    {
        const FOffgridAITextVisemeEvent& T = Plan.Events[NextEventIndex];
        const float OrderNorm = EventCenterNorms.IsValidIndex(NextEventIndex)
            ? EventCenterNorms[NextEventIndex]
            : 1.0f;
        const float RequiredActiveSec = OrderNorm * TotalPlannedActiveSec;

        if (RequiredActiveSec > CommitSafeActiveSec && !bFinal)
        {
            break;
        }

        float Center = 0.0f;
        const float MapActiveSec = FMath::Min(RequiredActiveSec, ObservedActiveSec);
        if (!MapActiveSpeechTimeToObservedClock(EffectiveRegions, MapActiveSec, Center))
        {
            if (!bFinal)
            {
                break;
            }
            Center = InOutTrack.SpeechStartSeconds + RequiredActiveSec;
        }

        const float Span = SpanForPose(T.PoseID);
        Center = FMath::Max(Center - LeadForPose(T.PoseID), 0.0f);

        if (NextEventIndex == 0 && FirstSpeechStart >= 0.0f)
        {
            Center = FMath::Max(Center, FirstSpeechStart + FMath::Min(Span * 0.42f, 0.060f));
        }
        if (LastCenter >= 0.0f)
        {
            Center = FMath::Max(Center, LastCenter + 0.050f);
        }
        if (!bFinal && Center - Input.CurrentPlaybackSec > MaxLiveLeadSec)
        {
            break;
        }

        FOffgridAIAlignedVisemeEvent E;
        FillEventFromText(Plan, NextEventIndex, T, OrderNorm, Center, Span, ObservedActiveSec, RequiredActiveSec, TotalPlannedActiveSec, Input, E);
        InOutTrack.Events.Add(E);
        LastCenter = Center;
        ++NextEventIndex;
    }

    // The LineCoach samples only when the committed track is marked built.
    // In the simplified occupancy runtime, the track is intentionally live and
    // prefix-built: as soon as any stable speech-occupancy events have been
    // committed, they are authoritative and must be available to the performer.
    // Waiting until final/full completion causes the whole front of each line to
    // be missed live, even though runtime_commit_events.csv contains early
    // committed centers.
    bInOutTrackBuilt = InOutTrack.Events.Num() > 0;
}
