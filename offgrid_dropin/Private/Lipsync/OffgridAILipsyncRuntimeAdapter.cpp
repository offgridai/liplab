#include "Lipsync/OffgridAILipsyncRuntimeAdapter.h"

namespace
{
static const FName OccupancyReason(TEXT("speech_occupancy_playhead"));
static const FName RegionClosedDropReason(TEXT("speech_region_closed_drop"));
static const FName MissingRegionDropReason(TEXT("speech_region_missing_drop"));
static constexpr float InterWordSpacerSeconds = 0.020f;
static constexpr float ActiveDurationScale = 0.90f;

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

// Layer 1 produces a duration prior, not absolute timestamps. These cumulative
// active seconds are prior-space durations that later get mapped onto observed
// speech-region time by the runtime.
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
        float Weight = FMath::Max(Phone.WeightSeconds * ActiveDurationScale, 0.018f);
        const int32 NextPhoneIndex = PhoneIndex + 1;
        OutPhoneStartActiveSeconds[PhoneIndex] = OutTotalActiveSeconds;
        OutPhoneCenterActiveSeconds[PhoneIndex] = OutTotalActiveSeconds + Weight * 0.5f;
        OutPhoneEndActiveSeconds[PhoneIndex] = OutTotalActiveSeconds + Weight;
        OutTotalActiveSeconds += Weight;

        if (Plan.ExpectedPhones.IsValidIndex(NextPhoneIndex))
        {
            const FOffgridAIExpectedPhone& NextPhone = Plan.ExpectedPhones[NextPhoneIndex];
            if (NextPhone.WordIndex != Phone.WordIndex)
            {
                OutTotalActiveSeconds += InterWordSpacerSeconds;
            }
        }
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

static bool IsSoftPausePunctuation(TCHAR C)
{
    return C == TEXT(',') || C == TEXT(';') || C == TEXT(':');
}

static bool IsHardPausePunctuation(TCHAR C)
{
    return C == TEXT('.') || C == TEXT('!') || C == TEXT('?');
}

static float HoldSecondsForBoundary(TCHAR C)
{
    if (IsSoftPausePunctuation(C))
    {
        return 0.120f;
    }
    if (IsHardPausePunctuation(C))
    {
        return 0.260f;
    }
    return 0.0f;
}

static float HoldSecondsForBoundary(TCHAR C, EOffgridAIBoundaryPauseClass PauseClass)
{
    switch (PauseClass)
    {
    case EOffgridAIBoundaryPauseClass::SoftListPause:
        return 0.450f;
    case EOffgridAIBoundaryPauseClass::HardBreakPause:
        return 1.200f;
    case EOffgridAIBoundaryPauseClass::None:
    default:
        return HoldSecondsForBoundary(C);
    }
}

static const FOffgridAIStreamingAudioFeatureFrame* FindFeatureFrameAtPlayback(
    const TArray<FOffgridAIStreamingAudioFeatureFrame>* AudioFeatureFrames,
    float PlaybackSec)
{
    if (!AudioFeatureFrames)
    {
        return nullptr;
    }

    for (int32 Index = AudioFeatureFrames->Num() - 1; Index >= 0; --Index)
    {
        const FOffgridAIStreamingAudioFeatureFrame& Frame = (*AudioFeatureFrames)[Index];
        if (PlaybackSec >= Frame.AudioBufferStartSec)
        {
            return &Frame;
        }
    }

    return AudioFeatureFrames->Num() > 0 ? &(*AudioFeatureFrames)[0] : nullptr;
}

static bool IsPunctuationPauseLullFrame(const FOffgridAIStreamingAudioFeatureFrame* Frame)
{
    if (!Frame)
    {
        return false;
    }

    return Frame->bStrongQuiet
        || Frame->bLowEvidence
        || Frame->PauseFamily == FName(TEXT("comma_lull"))
        || (Frame->RMSNorm <= 0.18f && Frame->SpeechEvidence <= 0.32f);
}

static bool IsPunctuationPauseSpeechFrame(const FOffgridAIStreamingAudioFeatureFrame* Frame)
{
    if (!Frame)
    {
        return false;
    }

    return Frame->bInSpeechAfterFrame
        && !Frame->bStrongQuiet
        && (Frame->SpeechEvidence >= 0.38f || Frame->RMSNorm >= 0.20f);
}

static int32 FindRegionIndexAtPlayback(const TArray<FEffectiveSpeechRegion>& Regions, float PlaybackSec)
{
    for (int32 RegionIndex = 0; RegionIndex < Regions.Num(); ++RegionIndex)
    {
        const FEffectiveSpeechRegion& Region = Regions[RegionIndex];
        if (PlaybackSec >= Region.StartSec && PlaybackSec <= Region.EndSec)
        {
            return RegionIndex;
        }
    }

    int32 BestIndex = INDEX_NONE;
    for (int32 RegionIndex = 0; RegionIndex < Regions.Num(); ++RegionIndex)
    {
        if (Regions[RegionIndex].StartSec <= PlaybackSec)
        {
            BestIndex = RegionIndex;
        }
        else
        {
            break;
        }
    }
    return BestIndex;
}

static void AdvancePlaybackHoldState(
    const FOffgridAILipsyncRuntimeUpdateInput& Input,
    const TArray<FEffectiveSpeechRegion>& EffectiveRegions,
    FOffgridAIPunctuationHoldState& InOutState)
{
    const float PlaybackSec = FMath::Max(Input.CurrentPlaybackSec, 0.0f);
    const float DeltaSec = InOutState.bPlayheadStarted
        ? FMath::Max(PlaybackSec - InOutState.LastPlaybackSec, 0.0f)
        : 0.0f;

    if (!InOutState.bPlayheadStarted && EffectiveRegions.Num() > 0)
    {
        InOutState.bPlayheadStarted = true;
        InOutState.PlaybackOriginSec = EffectiveRegions[0].StartSec;
        InOutState.LastPlaybackSec = PlaybackSec;
        InOutState.ActivePlayheadSec = FMath::Max(PlaybackSec - InOutState.PlaybackOriginSec, 0.0f);
        return;
    }

    if (!InOutState.bPlayheadStarted)
    {
        return;
    }

    auto ReanchorPausedClockToObservedResume = [&](float ResumeClockSec)
    {
        const float DesiredPausedSec =
            FMath::Max(ResumeClockSec - InOutState.PlaybackOriginSec - InOutState.ActivePlayheadSec, 0.0f);
        InOutState.TotalPausedSec = DesiredPausedSec;
    };

    if (InOutState.bHoldActive)
    {
        const FOffgridAIStreamingAudioFeatureFrame* PlaybackFrame =
            FindFeatureFrameAtPlayback(Input.AudioFeatureFrames, PlaybackSec);
        const bool bPauseTimeShouldAccumulate = InOutState.bObservedPauseLull;

        if (bPauseTimeShouldAccumulate)
        {
            InOutState.TotalPausedSec += DeltaSec;
        }

        InOutState.bObservedPauseLull =
            InOutState.bObservedPauseLull || IsPunctuationPauseLullFrame(PlaybackFrame);

        const float MinPauseHoldSec = 0.060f;
        const float HoldElapsedSec = PlaybackSec - InOutState.HoldStartPlaybackSec;
        if (InOutState.bObservedPauseLull
            && HoldElapsedSec >= MinPauseHoldSec
            && IsPunctuationPauseSpeechFrame(PlaybackFrame))
        {
            ReanchorPausedClockToObservedResume(PlaybackSec);
            InOutState.bHoldActive = false;
            InOutState.bObservedPauseLull = false;
            InOutState.ActivePauseClass = EOffgridAIBoundaryPauseClass::None;
        }
        else if (PlaybackSec >= InOutState.HoldDeadlinePlaybackSec)
        {
            InOutState.bHoldActive = false;
            InOutState.bObservedPauseLull = false;
            InOutState.ActivePauseClass = EOffgridAIBoundaryPauseClass::None;
        }
    }

    if (!InOutState.bHoldActive)
    {
        InOutState.ActivePlayheadSec += DeltaSec;
    }

    InOutState.LastPlaybackSec = PlaybackSec;
}

static void BuildWordStartSecondsFromPlaybackClock(
    const TArray<float>& WordStartActiveSeconds,
    const FOffgridAIPunctuationHoldState& HoldState,
    TArray<float>& OutWordStartSeconds)
{
    OutWordStartSeconds.Init(-1.0f, WordStartActiveSeconds.Num());
    if (!HoldState.bPlayheadStarted)
    {
        return;
    }

    const float PlaybackOffsetSec = HoldState.PlaybackOriginSec + HoldState.TotalPausedSec;
    for (int32 WordIndex = 0; WordIndex < WordStartActiveSeconds.Num(); ++WordIndex)
    {
        OutWordStartSeconds[WordIndex] = PlaybackOffsetSec + FMath::Max(WordStartActiveSeconds[WordIndex], 0.0f);
    }
}

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

static void FillDroppedEventFromText(
    int32 EventIndex,
    const FOffgridAITextVisemeEvent& T,
    float RegionStartSec,
    float RegionEndSec,
    float ObservedActiveSec,
    float RequiredActiveSec,
    const FOffgridAILipsyncRuntimeUpdateInput& Input,
    const FName DropReason,
    FOffgridAIDroppedVisemeEvent& Out)
{
    Out.EventIndex = EventIndex;
    Out.PoseID = T.PoseID;
    Out.SourceWord = T.SourceText;
    Out.WordIndex = T.WordIndex;
    Out.SpeechRegionIndex = T.SpeechRegionIndex;
    Out.SentenceIndex = T.SentenceIndex;
    Out.bIsStrongVisibleEvent = T.bIsStrongVisibleEvent;
    Out.SourcePhoneIndex = T.SourcePhoneGlobalIndex;
    Out.SourcePhoneBase = T.SourcePhoneBase;
    Out.SourcePhoneClass = SourcePhoneClassName(T.SourcePhoneBase);
    Out.DropPlaybackSeconds = Input.CurrentPlaybackSec;
    Out.RegionStartSeconds = RegionStartSec;
    Out.RegionEndSeconds = RegionEndSec;
    Out.RequiredActiveElapsedSeconds = RequiredActiveSec;
    Out.ObservedActiveElapsedSeconds = ObservedActiveSec;
    Out.DropReason = DropReason;
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
    const TArray<float>& WordStartSeconds,
    const FOffgridAILipsyncRuntimeUpdateInput& Input,
    FOffgridAIAlignedVisemeEvent& Out)
{
    Out.EventIndex = EventIndex;
    Out.PoseID = T.PoseID;
    Out.Strength = T.Strength;
    Out.SourceWord = T.SourceText;
    Out.WordIndex = T.WordIndex;
    Out.SentenceIndex = T.SentenceIndex;
    Out.SpeechRegionIndex = T.SpeechRegionIndex;
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

    if (WordStartSeconds.IsValidIndex(T.WordIndex) && WordStartSeconds[T.WordIndex] >= 0.0f)
    {
        Out.DetectedWordStartSeconds = WordStartSeconds[T.WordIndex];
        Out.bDetectedWordStartMappedToObservedSpeech = true;
    }
    else
    {
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
    RuntimeSpeechRegionDiagnosticRows.Reset();
    AudioOccupancyDiagnosticUpdateOrdinal = 0;
    StreamTailDiagnosticRow = FOffgridAIStreamTailDiagnosticRow();

    PCMChunkCount = 0;
    PCMBytesReceived = 0;
    PCMSamplesReceived = 0;
    LastPCMChunkSampleRate = 0;
    LastPCMChunkChannels = 0;
    LastPCMChunkStartSample = -1;
    LastPCMChunkEndSample = -1;
    PunctuationHoldState = FOffgridAIPunctuationHoldState();
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
    // Seal detector input at the currently observed audio end, but do not imply
    // that audible playback has finished. Callers must keep driving Update()
    // during buffered playback drain, and only call Finalize() at true playback end.
    Detector.Finalize(Detector.GetObservedAudioBufferEndSec());
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

    FOffgridAILipsyncRuntimeAdapter::UpdateCommittedTrack(Input, CommittedTrack, PunctuationHoldState, bCommittedTrackBuilt);
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

    FOffgridAILipsyncRuntimeAdapter::UpdateCommittedTrack(Input, CommittedTrack, PunctuationHoldState, bCommittedTrackBuilt);
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
    RuntimeSpeechRegionDiagnosticRows.Reset();
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

    for (const FOffgridAIStreamingSpeechRegion& SpeechRegion : ResolvedSpeechRegions)
    {
        FOffgridAIRuntimeSpeechRegionDiagnosticRow RegionRow;
        RegionRow.LineID = LineID;
        RegionRow.UpdateOrdinal = AudioOccupancyDiagnosticUpdateOrdinal;
        RegionRow.bFinalReplay = bFinalReplay;
        RegionRow.CurrentPlaybackSec = CurrentPlaybackSec;
        RegionRow.RegionIndex = SpeechRegion.SpeechRegionIndex;
        RegionRow.RegionOpenSec = SpeechRegion.AudioBufferStartSec;
        RegionRow.RegionCloseSec = SpeechRegion.AudioBufferEndSec;
        RegionRow.LastSpeechSec = SpeechRegion.AudioBufferLastSpeechSec;
        RegionRow.ProvisionalEndSec = SpeechRegion.ProvisionalEndSec;
        RegionRow.EndDecisionSec = SpeechRegion.EndDecisionSec;
        RegionRow.ReopenCount = SpeechRegion.ReopenCount;
        RegionRow.bStarted = SpeechRegion.bStarted;
        RegionRow.bEnded = SpeechRegion.bEnded;
        RegionRow.bContainsPlaybackSec =
            SpeechRegion.bStarted
            && CurrentPlaybackSec >= SpeechRegion.AudioBufferStartSec
            && CurrentPlaybackSec <= SpeechRegion.AudioBufferEndSec;
        RegionRow.CloseReason = SpeechRegion.EndReason;
        RegionRow.DiagnosticKind = FName(TEXT("runtime_speech_region"));

        for (const FOffgridAIAlignedVisemeEvent& E : CommittedTrack.Events)
        {
            if (E.SpeechRegionIndex == SpeechRegion.SpeechRegionIndex)
            {
                ++RegionRow.CommittedEventCount;
            }
        }
        for (const FOffgridAIDroppedVisemeEvent& E : CommittedTrack.DroppedEvents)
        {
            if (E.SpeechRegionIndex == SpeechRegion.SpeechRegionIndex)
            {
                ++RegionRow.DroppedEventCount;
            }
        }

        RuntimeSpeechRegionDiagnosticRows.Add(RegionRow);
    }
    ++AudioOccupancyDiagnosticUpdateOrdinal;
}

void FOffgridAILipsyncRuntimeAdapter::UpdateCommittedTrack(const FOffgridAILipsyncRuntimeUpdateInput& Input, FOffgridAIAlignedVisemeTrack& InOutTrack, FOffgridAIPunctuationHoldState& InOutHoldState, bool& bInOutTrackBuilt)
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

    const bool bStreamSealed = Input.bInputStreamClosed || Input.bPlaybackFinalized;
    const bool bPlaybackFinal = Input.bPlaybackFinalized;
    const float ObservedEnd = FMath::Max(Input.ObservedAudioBufferEndSec, 0.0f);

    TArray<FEffectiveSpeechRegion> EffectiveRegions;
    BuildEffectiveSpeechRegions(Input.SpeechRegions, ObservedEnd, bStreamSealed, EffectiveRegions);

    const float ObservedActiveSec = ComputeObservedActiveSpeechSeconds(EffectiveRegions);
    const float FirstSpeechStart = ComputeFirstSpeechStart(EffectiveRegions);
    const float LastSpeechEnd = ComputeLastSpeechEnd(EffectiveRegions, ObservedEnd);

    if (ObservedActiveSec <= 0.001f && !bStreamSealed)
    {
        return;
    }

    InOutTrack.SpeechStartSeconds = FirstSpeechStart >= 0.0f ? FirstSpeechStart : 0.0f;
    InOutTrack.SpeechEndSeconds = LastSpeechEnd;

    TArray<float> WordStartSeconds;
    AdvancePlaybackHoldState(Input, EffectiveRegions, InOutHoldState);
    BuildWordStartSecondsFromPlaybackClock(
        WordStartActiveSeconds,
        InOutHoldState,
        WordStartSeconds);

    // Runtime scheduling:
    // 1. transcript owns viseme identity and order,
    // 2. speech onset owns when playback may start,
    // 3. punctuation may open a bounded audio-aware hold,
    // 4. observed lull + resumed speech re-anchor the playhead,
    // 5. otherwise the hold expires and playback continues monotonically.
    const float CommitLagSec = bStreamSealed ? 0.0f : 0.030f;
    const float MinLiveLeadSec = bPlaybackFinal ? 0.0f : 0.040f;
    const float MaxLiveLeadSec = bStreamSealed ? 999.0f : FMath::Max(Input.PrerollSec + 0.120f, 0.250f);

    int32 NextEventIndex = InOutTrack.Events.Num();
    float LastCenter = InOutTrack.Events.Num() > 0 ? InOutTrack.Events.Last().FinalRenderCenterSeconds : -1.0f;
    const float TotalPlannedActiveSec = FMath::Max(TotalPhoneActiveSeconds, 0.001f);

    const float PlaybackOffsetSec =
        InOutHoldState.PlaybackOriginSec + InOutHoldState.TotalPausedSec;
    const float CommitSafeActiveSec = FMath::Max(InOutHoldState.ActivePlayheadSec - CommitLagSec, 0.0f);

    while (Plan.Events.IsValidIndex(NextEventIndex))
    {
        const FOffgridAITextVisemeEvent& T = Plan.Events[NextEventIndex];
        if (!InOutHoldState.bPlayheadStarted || EffectiveRegions.Num() <= 0)
        {
            break;
        }

        if (InOutHoldState.bHoldActive)
        {
            break;
        }

        const bool bAtWordStart = Plan.WordVisibleEventBeginIndices.IsValidIndex(T.WordIndex)
            && Plan.WordVisibleEventBeginIndices[T.WordIndex] == NextEventIndex;
        if (bAtWordStart && T.WordIndex > 0 && !InOutHoldState.bHoldActive)
        {
            const int32 BoundaryWordIndex = T.WordIndex - 1;
            const TCHAR Boundary = Plan.WordBoundaryPunctuationAfter.IsValidIndex(BoundaryWordIndex)
                ? Plan.WordBoundaryPunctuationAfter[BoundaryWordIndex]
                : TCHAR(0);
            const EOffgridAIBoundaryPauseClass BoundaryPauseClass =
                Plan.WordBoundaryPauseClassAfter.IsValidIndex(BoundaryWordIndex)
                    ? Plan.WordBoundaryPauseClassAfter[BoundaryWordIndex]
                    : EOffgridAIBoundaryPauseClass::None;
            const float HoldSeconds = HoldSecondsForBoundary(Boundary, BoundaryPauseClass);
            if (HoldSeconds > 0.0f && InOutHoldState.BoundaryWordIndex != BoundaryWordIndex)
            {
                InOutHoldState.bHoldActive = true;
                InOutHoldState.bObservedPauseLull = false;
                InOutHoldState.BoundaryWordIndex = BoundaryWordIndex;
                InOutHoldState.ActivePauseClass = BoundaryPauseClass;
                InOutHoldState.HoldStartPlaybackSec = Input.CurrentPlaybackSec;
                InOutHoldState.HoldDeadlinePlaybackSec = Input.CurrentPlaybackSec + HoldSeconds;
                break;
            }
        }

        const float OrderNorm = EventCenterNorms.IsValidIndex(NextEventIndex)
            ? EventCenterNorms[NextEventIndex]
            : 1.0f;
        const int32 SourcePhoneGlobalIndex = T.SourcePhoneGlobalIndex;
        const float RequiredActiveSec = PhoneCenterActiveSeconds.IsValidIndex(SourcePhoneGlobalIndex)
            ? FMath::Max(PhoneCenterActiveSeconds[SourcePhoneGlobalIndex], 0.0f)
            : OrderNorm * TotalPlannedActiveSec;

        if (RequiredActiveSec > CommitSafeActiveSec && !bStreamSealed)
        {
            break;
        }

        const float RequiredPhoneStartActiveSec = PhoneStartActiveSeconds.IsValidIndex(SourcePhoneGlobalIndex)
            ? FMath::Max(PhoneStartActiveSeconds[SourcePhoneGlobalIndex], 0.0f)
            : FMath::Max(RequiredActiveSec - 0.040f, 0.0f);
        const float RequiredPhoneEndActiveSec = PhoneEndActiveSeconds.IsValidIndex(SourcePhoneGlobalIndex)
            ? FMath::Max(PhoneEndActiveSeconds[SourcePhoneGlobalIndex], 0.0f)
            : (RequiredActiveSec + 0.040f);

        float BaseStart = PlaybackOffsetSec + RequiredPhoneStartActiveSec;
        float Center = PlaybackOffsetSec + RequiredActiveSec;
        float BaseEnd = PlaybackOffsetSec + RequiredPhoneEndActiveSec;
        const float PriorCenter = Center;

        const float BaseSpan = FMath::Max(BaseEnd - BaseStart, 0.020f);
        const float Span = FMath::Max(SpanForPose(T.PoseID), BaseSpan);
        Center = FMath::Max(Center - LeadForPose(T.PoseID), 0.0f);
        const float LeadAdjustedCenter = Center;
        float MinLiveLeadDelay = 0.0f;
        float InterEventFloorDelay = 0.0f;

        if (RequiredActiveSec <= 0.001f)
        {
            Center = FMath::Max(Center, PlaybackOffsetSec + FMath::Min(Span * 0.42f, 0.060f));
        }
        if (!bPlaybackFinal)
        {
            const float MinAllowedCenter = Input.CurrentPlaybackSec + MinLiveLeadSec;
            if (Center < MinAllowedCenter)
            {
                MinLiveLeadDelay = MinAllowedCenter - Center;
                Center = MinAllowedCenter;
            }
        }
        if (LastCenter >= 0.0f)
        {
            const float MinSpacingCenter = LastCenter + 0.050f;
            if (Center < MinSpacingCenter)
            {
                InterEventFloorDelay = MinSpacingCenter - Center;
                Center = MinSpacingCenter;
            }
        }
        if (!bPlaybackFinal && Center - Input.CurrentPlaybackSec > MaxLiveLeadSec)
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
            InOutHoldState.ActivePlayheadSec,
            RequiredActiveSec,
            TotalPlannedActiveSec,
            EffectiveRegions,
            WordStartActiveSeconds,
            WordStartSeconds,
            Input,
            E);
        E.PriorStartSeconds = BaseStart;
        E.PriorCenterSeconds = PriorCenter;
        E.PriorEndSeconds = BaseEnd;
        E.LeadAdjustedCenterSeconds = LeadAdjustedCenter;
        E.PlaybackOffsetSeconds = PlaybackOffsetSec;
        E.TotalPausedSecondsAtCommit = InOutHoldState.TotalPausedSec;
        E.MinLiveLeadDelaySeconds = MinLiveLeadDelay;
        E.InterEventFloorDelaySeconds = InterEventFloorDelay;
        E.TotalCenterDelaySeconds = FMath::Max(Center - PriorCenter, 0.0f);
        E.SpeechRegionIndex = FMath::Max(FindRegionIndexAtPlayback(EffectiveRegions, Input.CurrentPlaybackSec), 0);
        E.CommitReason = OccupancyReason;
        if (LastCenter >= 0.0f && E.FinalRenderCenterSeconds < LastCenter + 0.001f)
        {
            float ForcedCenter = LastCenter + 0.001f;
            if (ForcedCenter < 0.0f)
            {
                ForcedCenter = 0.0f;
            }
            E.FinalRenderCenterSeconds = ForcedCenter;
        }
        E.RenderStartSeconds = FMath::Min(E.RenderStartSeconds, E.FinalRenderCenterSeconds);
        E.RenderEndSeconds = FMath::Max(E.RenderEndSeconds, E.FinalRenderCenterSeconds);
        InOutTrack.Events.Add(E);
        LastCenter = E.FinalRenderCenterSeconds;
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
