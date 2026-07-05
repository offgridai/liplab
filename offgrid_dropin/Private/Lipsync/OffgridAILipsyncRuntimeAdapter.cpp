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

static void BuildPhoneActiveTimings(
    const FOffgridAITextVisemePlan& Plan,
    TArray<float>& OutPhoneStartActiveSeconds,
    TArray<float>& OutPhoneCenterActiveSeconds,
    TArray<float>& OutPhoneEndActiveSeconds,
    float& OutTotalActiveSeconds)
{
    const int32 PhoneCount = Plan.ExpectedPhones.Num();
    OutPhoneStartActiveSeconds.Init(0.0f, PhoneCount);
    OutPhoneCenterActiveSeconds.Init(0.0f, PhoneCount);
    OutPhoneEndActiveSeconds.Init(0.0f, PhoneCount);
    OutTotalActiveSeconds = 0.0f;
    if (PhoneCount <= 0)
    {
        return;
    }

    for (int32 PhoneIndex = 0; PhoneIndex < Plan.ExpectedPhones.Num(); ++PhoneIndex)
    {
        const FOffgridAIExpectedPhone& Phone = Plan.ExpectedPhones[PhoneIndex];
        const float Weight = FMath::Max(Phone.WeightSeconds, 0.020f);
        OutPhoneStartActiveSeconds[PhoneIndex] = OutTotalActiveSeconds;
        OutPhoneCenterActiveSeconds[PhoneIndex] = OutTotalActiveSeconds + Weight * 0.5f;
        OutPhoneEndActiveSeconds[PhoneIndex] = OutTotalActiveSeconds + Weight;
        OutTotalActiveSeconds += Weight;
    }
    OutTotalActiveSeconds = FMath::Max(OutTotalActiveSeconds, 0.001f);
}

static void BuildEventProgressNormsFromPhones(
    const FOffgridAITextVisemePlan& Plan,
    const TArray<float>& PhoneCenterActiveSeconds,
    float TotalPhoneActiveSeconds,
    TArray<float>& OutCenterNorms)
{
    OutCenterNorms.Init(0.0f, Plan.Events.Num());
    const float SafeTotal = FMath::Max(TotalPhoneActiveSeconds, 0.001f);
    for (int32 EventIndex = 0; EventIndex < Plan.Events.Num(); ++EventIndex)
    {
        const FOffgridAITextVisemeEvent& Event = Plan.Events[EventIndex];
        if (PhoneCenterActiveSeconds.IsValidIndex(Event.SourcePhoneGlobalIndex))
        {
            OutCenterNorms[EventIndex] = FMath::Clamp(
                PhoneCenterActiveSeconds[Event.SourcePhoneGlobalIndex] / SafeTotal,
                0.0f,
                1.0f);
        }
    }
}

static void BuildWordStartActiveSeconds(
    const FOffgridAITextVisemePlan& Plan,
    const TArray<float>& PhoneStartActiveSeconds,
    TArray<float>& OutWordStartActiveSeconds)
{
    OutWordStartActiveSeconds.Init(0.0f, Plan.WordSyllableCounts.Num());
    if (Plan.WordSyllableCounts.Num() <= 0)
    {
        return;
    }

    for (int32 WordIndex = 0; WordIndex < Plan.WordSyllableCounts.Num(); ++WordIndex)
    {
        if (!OutWordStartActiveSeconds.IsValidIndex(WordIndex))
        {
            continue;
        }
        if (Plan.WordPhoneBeginIndices.IsValidIndex(WordIndex))
        {
            const int32 PhoneBegin = Plan.WordPhoneBeginIndices[WordIndex];
            if (PhoneStartActiveSeconds.IsValidIndex(PhoneBegin))
            {
                OutWordStartActiveSeconds[WordIndex] = PhoneStartActiveSeconds[PhoneBegin];
                continue;
            }
        }
    }

    if (Plan.ExpectedPhones.Num() > 0)
    {
        return;
    }

    float FallbackCursor = 0.0f;
    for (int32 WordIndex = 0; WordIndex < Plan.WordSyllableCounts.Num(); ++WordIndex)
    {
        OutWordStartActiveSeconds[WordIndex] = FallbackCursor;
        FallbackCursor += 0.075f * FMath::Max(Plan.WordSyllableCounts[WordIndex], 1);
    }
}

struct FEffectiveSpeechRegion
{
    float StartSec = 0.0f;
    float EndSec = 0.0f;
};

struct FPlannedSpeechRegionActiveSpan
{
    float StartActiveSec = 0.0f;
    float EndActiveSec = 0.0f;
    bool bValid = false;
};

static float SpeechRegionObservedEnd(const FOffgridAIStreamingSpeechRegion& SpeechRegion, float ObservedEndSec, bool bFinal)
{
    if (SpeechRegion.bEnded || bFinal)
    {
        return FMath::Clamp(SpeechRegion.AudioBufferEndSec, SpeechRegion.AudioBufferStartSec, ObservedEndSec);
    }
    return FMath::Max(SpeechRegion.AudioBufferStartSec, ObservedEndSec);
}

static void BuildEffectiveSpeechRegions(
    const TArray<FOffgridAIStreamingSpeechRegion>* SpeechRegions,
    const TArray<FOffgridAIStreamingAudioFeatureFrame>* AudioFeatureFrames,
    float ObservedEndSec,
    bool bFinal,
    TArray<FEffectiveSpeechRegion>& OutRegions)
{
    OutRegions.Reset();
    if (!SpeechRegions) return;

    for (const FOffgridAIStreamingSpeechRegion& SpeechRegion : *SpeechRegions)
    {
        if (!SpeechRegion.bStarted) continue;

        const float S = FMath::Clamp(SpeechRegion.AudioBufferStartSec, 0.0f, ObservedEndSec);
        const float E = FMath::Clamp(SpeechRegionObservedEnd(SpeechRegion, ObservedEndSec, bFinal), S, FMath::Max(ObservedEndSec, S));
        if (E - S <= 0.010f) continue;
        FEffectiveSpeechRegion Region;
        Region.StartSec = S;
        Region.EndSec = E;
        OutRegions.Add(Region);
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

static float ComputeObservedRegionActiveSeconds(const TArray<FEffectiveSpeechRegion>& Regions, int32 RegionIndex)
{
    return Regions.IsValidIndex(RegionIndex)
        ? FMath::Max(Regions[RegionIndex].EndSec - Regions[RegionIndex].StartSec, 0.0f)
        : 0.0f;
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

static bool MapRegionLocalActiveToObservedClock(
    const TArray<FEffectiveSpeechRegion>& Regions,
    int32 RegionIndex,
    float TargetLocalActiveSec,
    float& OutClockSec)
{
    if (!Regions.IsValidIndex(RegionIndex))
    {
        return false;
    }

    const FEffectiveSpeechRegion& Region = Regions[RegionIndex];
    const float RegionDur = FMath::Max(Region.EndSec - Region.StartSec, 0.0f);
    OutClockSec = Region.StartSec + FMath::Clamp(TargetLocalActiveSec, 0.0f, RegionDur);
    return true;
}

static void BuildPlannedSpeechRegionActiveSpans(
    const FOffgridAITextVisemePlan& Plan,
    const TArray<float>& PhoneStartActiveSeconds,
    const TArray<float>& PhoneEndActiveSeconds,
    TArray<FPlannedSpeechRegionActiveSpan>& OutSpans)
{
    int32 RegionCount = 0;
    for (const FOffgridAIExpectedPhone& Phone : Plan.ExpectedPhones)
    {
        RegionCount = FMath::Max(RegionCount, Phone.SentenceIndex + 1);
    }
    for (const int32 RegionIndex : Plan.WordSentenceIndices)
    {
        RegionCount = FMath::Max(RegionCount, RegionIndex + 1);
    }

    OutSpans.Init(FPlannedSpeechRegionActiveSpan(), RegionCount);
    for (int32 PhoneIndex = 0; PhoneIndex < Plan.ExpectedPhones.Num(); ++PhoneIndex)
    {
        const FOffgridAIExpectedPhone& Phone = Plan.ExpectedPhones[PhoneIndex];
        if (!OutSpans.IsValidIndex(Phone.SentenceIndex)
            || !PhoneStartActiveSeconds.IsValidIndex(PhoneIndex)
            || !PhoneEndActiveSeconds.IsValidIndex(PhoneIndex))
        {
            continue;
        }

        FPlannedSpeechRegionActiveSpan& Span = OutSpans[Phone.SentenceIndex];
        const float PhoneStart = PhoneStartActiveSeconds[PhoneIndex];
        const float PhoneEnd = PhoneEndActiveSeconds[PhoneIndex];
        if (!Span.bValid)
        {
            Span.StartActiveSec = PhoneStart;
            Span.EndActiveSec = PhoneEnd;
            Span.bValid = true;
            continue;
        }

        Span.StartActiveSec = FMath::Min(Span.StartActiveSec, PhoneStart);
        Span.EndActiveSec = FMath::Max(Span.EndActiveSec, PhoneEnd);
    }

    float Cursor = 0.0f;
    for (FPlannedSpeechRegionActiveSpan& Span : OutSpans)
    {
        if (!Span.bValid)
        {
            Span.StartActiveSec = Cursor;
            Span.EndActiveSec = Cursor;
            continue;
        }
        Cursor = FMath::Max(Cursor, Span.EndActiveSec);
    }
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
    const TArray<FEffectiveSpeechRegion>& EffectiveRegions,
    const TArray<float>& WordStartActiveSeconds,
    const FOffgridAILipsyncRuntimeUpdateInput& Input,
    FOffgridAIAlignedVisemeEvent& Out)
{
    Out.EventIndex = EventIndex;
    Out.PoseID = T.PoseID;
    Out.Strength = T.Strength;
    Out.SourceWord = T.SourceText;
    Out.WordIndex = T.WordIndex;
    Out.SentenceIndex = T.SentenceIndex;
    Out.SpeechRegionIndex = T.SentenceIndex;
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

    float WordStartClockSec = 0.0f;
    const float WordStartActiveSec = WordStartActiveSeconds.IsValidIndex(T.WordIndex) ? WordStartActiveSeconds[T.WordIndex] : 0.0f;
    if (MapActiveSpeechTimeToObservedClock(EffectiveRegions, WordStartActiveSec, WordStartClockSec))
    {
        Out.DetectedWordStartSeconds = WordStartClockSec;
        Out.bDetectedWordStartMappedToObservedSpeech = true;
    }
    else if (EffectiveRegions.Num() > 0)
    {
        Out.DetectedWordStartSeconds = EffectiveRegions[0].StartSec;
        Out.bDetectedWordStartMappedToObservedSpeech = false;
    }
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
    ResolvedSpeechRegions.Reset();
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
    RefreshResolvedSpeechRegions();
    CommittedTrack.NPCID = NPCID;
    CommittedTrack.LineID = LineID;
    bBegun = true;
}

void FOffgridAILipsyncRuntimeSession::PushAudioPCM16(const TArray<uint8>& PCMChunk, int32 BytesToUse, int32 SampleRate, int32 NumChannels, int64 ChunkStartSample)
{
    if (!bBegun) return;

    Detector.AppendPCM16(PCMChunk, BytesToUse, SampleRate, NumChannels, ChunkStartSample);
    RefreshResolvedSpeechRegions();

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
    RefreshResolvedSpeechRegions();
}

void FOffgridAILipsyncRuntimeSession::Update(float CurrentPlaybackSec)
{
    PlaybackSec = FMath::Max(CurrentPlaybackSec, 0.0f);
    UpdatePlaybackGate(Detector.GetObservedAudioBufferEndSec());
    RefreshResolvedSpeechRegions();

    FOffgridAILipsyncRuntimeUpdateInput Input;
    Input.TextPlan = &TextPlan;
    Input.SpeechRegions = &ResolvedSpeechRegions;
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
    RefreshResolvedSpeechRegions();

    FOffgridAILipsyncRuntimeUpdateInput Input;
    Input.TextPlan = &TextPlan;
    Input.SpeechRegions = &ResolvedSpeechRegions;
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
    const bool bHasSpeech = Detector.HasObservedFirstSpeechStart() || ResolvedSpeechRegions.Num() > 0;
    if (bHasSpeech || ObservedEndSec >= PrerollSec || bInputStreamClosed)
    {
        bPlaybackStarted = true;
    }
}

void FOffgridAILipsyncRuntimeSession::RefreshResolvedSpeechRegions()
{
    ResolvedSpeechRegions = Detector.GetSpeechRegions();
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
    StreamTailDiagnosticRow.SpeechRegionCount = Detector.GetSpeechRegions().Num();
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

    TArray<float> PhoneStartActiveSeconds;
    TArray<float> PhoneCenterActiveSeconds;
    TArray<float> PhoneEndActiveSeconds;
    float TotalPhoneActiveSeconds = 0.0f;
    BuildPhoneActiveTimings(
        Plan,
        PhoneStartActiveSeconds,
        PhoneCenterActiveSeconds,
        PhoneEndActiveSeconds,
        TotalPhoneActiveSeconds);

    TArray<float> EventCenterNorms;
    BuildEventProgressNormsFromPhones(Plan, PhoneCenterActiveSeconds, TotalPhoneActiveSeconds, EventCenterNorms);
    TArray<float> WordStartActiveSeconds;
    BuildWordStartActiveSeconds(Plan, PhoneStartActiveSeconds, WordStartActiveSeconds);

    const bool bFinal = Input.bInputStreamClosed || Input.bPlaybackFinalized;
    const float ObservedEnd = FMath::Max(Input.ObservedAudioBufferEndSec, 0.0f);

    TArray<FEffectiveSpeechRegion> EffectiveRegions;
    BuildEffectiveSpeechRegions(Input.SpeechRegions, Input.AudioFeatureFrames, ObservedEnd, bFinal, EffectiveRegions);

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
    const float CommitLagSec = bFinal ? 0.0f : 0.030f;
    const float MinLiveLeadSec = bFinal ? 0.0f : 0.040f;
    const float PhraseWarmupLeadSec = bFinal ? 0.0f : 0.060f;
    const float PhraseWarmupWindowSec = bFinal ? 0.0f : 0.180f;
    const float MaxLiveLeadSec = bFinal ? 999.0f : FMath::Max(Input.PrerollSec + 0.120f, 0.250f);

    int32 MaxTimingGroupIndex = -1;
    for (const FOffgridAITextVisemeEvent& Event : Plan.Events)
    {
        MaxTimingGroupIndex = FMath::Max(MaxTimingGroupIndex, Event.SentenceIndex);
    }
    TArray<int32> GroupFirstEventIndices;
    GroupFirstEventIndices.Init(INDEX_NONE, MaxTimingGroupIndex + 1);
    for (int32 EventIndex = 0; EventIndex < Plan.Events.Num(); ++EventIndex)
    {
        const int32 TimingGroupIndex = Plan.Events[EventIndex].SentenceIndex;
        if (GroupFirstEventIndices.IsValidIndex(TimingGroupIndex) && GroupFirstEventIndices[TimingGroupIndex] == INDEX_NONE)
        {
            GroupFirstEventIndices[TimingGroupIndex] = EventIndex;
        }
    }

    int32 NextEventIndex = InOutTrack.Events.Num();
    float LastCenter = InOutTrack.Events.Num() > 0 ? InOutTrack.Events.Last().FinalRenderCenterSeconds : -1.0f;

    while (Plan.Events.IsValidIndex(NextEventIndex))
    {
        const FOffgridAITextVisemeEvent& T = Plan.Events[NextEventIndex];
        if (EffectiveRegions.Num() <= 0)
        {
            break;
        }

        const float CommitSafeRegionActiveSec = FMath::Max(ObservedActiveSec - CommitLagSec, 0.0f);
        const float TotalPlannedRegionActiveSec = bFinal
            ? FMath::Max(ObservedActiveSec, 0.001f)
            : FMath::Max(TotalPhoneActiveSeconds, ObservedActiveSec);
        const float OrderNorm = EventCenterNorms.IsValidIndex(NextEventIndex)
            ? EventCenterNorms[NextEventIndex]
            : 1.0f;
        const int32 SourcePhoneGlobalIndex = T.SourcePhoneGlobalIndex;
        const auto ComputeRequiredRegionLocalActiveSec = [&](
            int32 EventIndex,
            int32 InSourcePhoneGlobalIndex,
            float InOrderNorm) -> float
        {
            return PhoneCenterActiveSeconds.IsValidIndex(InSourcePhoneGlobalIndex)
                ? FMath::Max(PhoneCenterActiveSeconds[InSourcePhoneGlobalIndex], 0.0f)
                : (InOrderNorm * TotalPlannedRegionActiveSec);
        };
        const float RequiredRegionLocalActiveSec = ComputeRequiredRegionLocalActiveSec(
            NextEventIndex,
            SourcePhoneGlobalIndex,
            OrderNorm);

        const bool bPhraseAlreadyStarted = (
            InOutTrack.Events.Num() > 0
            && InOutTrack.Events.Last().SentenceIndex == T.SentenceIndex
        );
        float SentenceAnchorCenter = 0.0f;
        float SentenceAnchorRequiredActiveSec = 0.0f;
        bool bSentenceAnchorFound = false;
        if (bPhraseAlreadyStarted)
        {
            for (const FOffgridAIAlignedVisemeEvent& ExistingEvent : InOutTrack.Events)
            {
                if (ExistingEvent.SentenceIndex != T.SentenceIndex)
                {
                    continue;
                }
                SentenceAnchorCenter = ExistingEvent.FinalRenderCenterSeconds;
                SentenceAnchorRequiredActiveSec = ExistingEvent.RequiredActiveElapsedSeconds;
                bSentenceAnchorFound = true;
                break;
            }
        }

        float FirstSentenceRequiredRegionLocalActiveSec = RequiredRegionLocalActiveSec;
        if (GroupFirstEventIndices.IsValidIndex(T.SentenceIndex))
        {
            const int32 FirstSentenceEventIndex = GroupFirstEventIndices[T.SentenceIndex];
            if (Plan.Events.IsValidIndex(FirstSentenceEventIndex))
            {
                const FOffgridAITextVisemeEvent& FirstSentenceEvent = Plan.Events[FirstSentenceEventIndex];
                const float FirstSentenceOrderNorm = EventCenterNorms.IsValidIndex(FirstSentenceEventIndex)
                    ? EventCenterNorms[FirstSentenceEventIndex]
                    : 0.0f;
                FirstSentenceRequiredRegionLocalActiveSec = ComputeRequiredRegionLocalActiveSec(
                    FirstSentenceEventIndex,
                    FirstSentenceEvent.SourcePhoneGlobalIndex,
                    FirstSentenceOrderNorm);
            }
        }

        const bool bPhraseWarmupEligible = (
            !bFinal
            && !bPhraseAlreadyStarted
            && ObservedActiveSec > 0.0f
            && (RequiredRegionLocalActiveSec - FirstSentenceRequiredRegionLocalActiveSec) <= PhraseWarmupWindowSec
        );

        if (RequiredRegionLocalActiveSec > CommitSafeRegionActiveSec && !bFinal && !bPhraseWarmupEligible)
        {
            break;
        }

        float BaseStart = 0.0f;
        float Center = 0.0f;
        float BaseEnd = 0.0f;
        const float RequiredPhoneStartRegionLocalActiveSec = PhoneStartActiveSeconds.IsValidIndex(SourcePhoneGlobalIndex)
            ? FMath::Max(PhoneStartActiveSeconds[SourcePhoneGlobalIndex], 0.0f)
            : FMath::Max(RequiredRegionLocalActiveSec - 0.040f, 0.0f);
        const float RequiredPhoneEndRegionLocalActiveSec = PhoneEndActiveSeconds.IsValidIndex(SourcePhoneGlobalIndex)
            ? FMath::Max(PhoneEndActiveSeconds[SourcePhoneGlobalIndex], 0.0f)
            : (RequiredRegionLocalActiveSec + 0.040f);

        const float MapStartActiveSec = FMath::Min(RequiredPhoneStartRegionLocalActiveSec, ObservedActiveSec);
        const float MapActiveSec = FMath::Min(RequiredRegionLocalActiveSec, ObservedActiveSec);
        const float MapEndActiveSec = FMath::Min(RequiredPhoneEndRegionLocalActiveSec, ObservedActiveSec);
        const bool bMappedStart = MapActiveSpeechTimeToObservedClock(EffectiveRegions, MapStartActiveSec, BaseStart);
        const bool bMappedCenter = MapActiveSpeechTimeToObservedClock(EffectiveRegions, MapActiveSec, Center);
        const bool bMappedEnd = MapActiveSpeechTimeToObservedClock(EffectiveRegions, MapEndActiveSec, BaseEnd);
        if (!bMappedCenter)
        {
            if (!bFinal)
            {
                break;
            }
            Center = EffectiveRegions[0].StartSec + RequiredRegionLocalActiveSec;
        }
        if (!bMappedStart)
        {
            BaseStart = FMath::Max(Center - 0.040f, 0.0f);
        }
        if (!bMappedEnd)
        {
            BaseEnd = Center + 0.040f;
        }

        const float BaseSpan = FMath::Max(BaseEnd - BaseStart, 0.020f);
        const float Span = FMath::Max(SpanForPose(T.PoseID), BaseSpan);
        Center = FMath::Max(Center - LeadForPose(T.PoseID), 0.0f);

        if (NextEventIndex == 0 && FirstSpeechStart >= 0.0f)
        {
            Center = FMath::Max(Center, FirstSpeechStart + FMath::Min(Span * 0.42f, 0.060f));
        }
        if (LastCenter >= 0.0f)
        {
            Center = FMath::Max(Center, LastCenter + 0.050f);
        }
        if (!bFinal)
        {
            if (!bSentenceAnchorFound && bPhraseWarmupEligible)
            {
                const float WarmStartCenter = Input.CurrentPlaybackSec + PhraseWarmupLeadSec;
                Center = FMath::Max(Center, WarmStartCenter);
                SentenceAnchorCenter = Center;
                SentenceAnchorRequiredActiveSec = RequiredRegionLocalActiveSec;
                bSentenceAnchorFound = true;
            }
            else if (bSentenceAnchorFound && RequiredRegionLocalActiveSec >= SentenceAnchorRequiredActiveSec)
            {
                // Once a sentence group has entered live playback, preserve its internal
                // pacing from that anchored onset rather than letting later
                // historical commits collapse into a single playback moment.
                const float SentenceRelativeCenter = SentenceAnchorCenter + (RequiredRegionLocalActiveSec - SentenceAnchorRequiredActiveSec);
                Center = FMath::Max(Center, SentenceRelativeCenter);
            }

            // Live playback is monotonic. If a commit arrives after its ideal
            // historical center has already passed, reschedule it slightly ahead
            // of the current playback cursor instead of emitting an already-late
            // event that can never render in time.
            Center = FMath::Max(Center, Input.CurrentPlaybackSec + MinLiveLeadSec);
        }
        if (!bFinal && Center - Input.CurrentPlaybackSec > MaxLiveLeadSec)
        {
            break;
        }

        FOffgridAIAlignedVisemeEvent E;
        FillEventFromText(
            Plan,
            NextEventIndex,
            T,
            OrderNorm,
            Center,
            Span,
            ObservedActiveSec,
            RequiredRegionLocalActiveSec,
            TotalPlannedRegionActiveSec,
            EffectiveRegions,
            WordStartActiveSeconds,
            Input,
            E);
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
