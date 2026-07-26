#include "Lipsync/OffgridAILipsyncRuntimeAdapter.h"
#include "Lipsync/OffgridAILipsyncVersion.h"

#include "Lipsync/OffgridAIStreamingEvidenceSurface.h"
#include "Lipsync/OffgridAIStreamingSyllablePositionEstimator.h"

static void UpdateSyllablePacedVisemeTrack(
    const FOffgridAILipsyncRuntimeUpdateInput& Input,
    FOffgridAICommittedVisemeTrack& InOutTrack,
    FOffgridAILipsyncSchedulerState& State,
    bool& bInOutTrackBuilt);

namespace
{
static constexpr float InterWordSeconds = 0.020f;
static constexpr float MinLiveLeadSec = 0.030f;
static constexpr float PulseCommitStabilitySec = 0.120f;
static constexpr float SchedulerEvidenceHistorySec = 0.250f;
static constexpr float MinimumAdaptiveWordPriorRate = 0.65f;
static constexpr float MaximumAdaptiveWordPriorRate = 1.65f;
static constexpr float AdaptiveWordPriorRateBlend = 0.35f;
static constexpr float MinimumWordRateObservationSec = 0.080f;
static constexpr float AmbiguousWordHeadLookaheadSec = 0.300f;

static float SpanForPose(const FName& PoseID)
{
    const FString Pose = PoseID.ToString();
    if (Pose.Contains(TEXT("22_MBP"))) return 0.110f;
    if (Pose.Contains(TEXT("20_FV"))) return 0.105f;
    if (Pose.Contains(TEXT("12_Ww")) || Pose.Contains(TEXT("11_Oo"))
        || Pose.Contains(TEXT("09_Oh"))) return 0.125f;
    return 0.135f;
}

static FName PhoneClass(const FString& Phone)
{
    if (Phone == TEXT("M") || Phone == TEXT("B") || Phone == TEXT("P")) return FName(TEXT("mbp"));
    if (Phone == TEXT("F") || Phone == TEXT("V")) return FName(TEXT("fv"));
    if (Phone == TEXT("W") || Phone == TEXT("Y")) return FName(TEXT("glide"));
    if (Phone == TEXT("S") || Phone == TEXT("Z") || Phone == TEXT("SH")
        || Phone == TEXT("ZH") || Phone == TEXT("CH") || Phone == TEXT("JH"))
        return FName(TEXT("sibilant"));
    if (Phone == TEXT("AA") || Phone == TEXT("AE") || Phone == TEXT("AH")
        || Phone == TEXT("AW") || Phone == TEXT("AY"))
        return FName(TEXT("open"));
    if (Phone == TEXT("EH") || Phone == TEXT("ER") || Phone == TEXT("EY")
        || Phone == TEXT("IH") || Phone == TEXT("IY"))
        return FName(TEXT("front"));
    if (Phone == TEXT("AO") || Phone == TEXT("OW") || Phone == TEXT("OY")
        || Phone == TEXT("UH") || Phone == TEXT("UW"))
        return FName(TEXT("round"));
    return FName(TEXT("other"));
}

static int32 RegionContaining(
    const TArray<FOffgridAIStreamingSpeechRegion>& Regions,
    float TimeSec);

static bool VisualLandmarkForPhone(
    const FString& Phone,
    EOffgridAIAudioLandmarkType& OutType,
    float& OutExpectedLeadSec,
    float& OutVisualLeadSec)
{
    if (Phone == TEXT("M") || Phone == TEXT("B") || Phone == TEXT("P"))
    {
        OutType = EOffgridAIAudioLandmarkType::Bilabial;
        OutExpectedLeadSec = 0.100f;
        OutVisualLeadSec = 0.025f;
        return true;
    }
    if (Phone == TEXT("F") || Phone == TEXT("V"))
    {
        OutType = EOffgridAIAudioLandmarkType::Labiodental;
        OutExpectedLeadSec = 0.100f;
        OutVisualLeadSec = 0.010f;
        return true;
    }
    if (Phone == TEXT("S") || Phone == TEXT("Z")
        || Phone == TEXT("SH") || Phone == TEXT("ZH")
        || Phone == TEXT("CH") || Phone == TEXT("JH"))
    {
        OutType = EOffgridAIAudioLandmarkType::Sibilant;
        OutExpectedLeadSec = 0.100f;
        OutVisualLeadSec = 0.010f;
        return true;
    }
    if (Phone == TEXT("W") || Phone == TEXT("Y"))
    {
        OutType = EOffgridAIAudioLandmarkType::Glide;
        OutExpectedLeadSec = 0.070f;
        OutVisualLeadSec = 0.015f;
        return true;
    }
    return false;
}

static float ResolveTranscriptConditionedVisualAnchor(
    const FOffgridAITextVisemePlan& Plan,
    int32 AnchorEventIndex,
    const TArray<FOffgridAIAudioLandmarkObservation>& Evidence,
    const TArray<FOffgridAIStreamingSpeechRegion>& SpeechRegions,
    int32 AudioRegionIndex,
    float NucleusCenterSec,
    float PreviousNucleusCenterSec,
    float ObservedAudioBufferEndSec,
    FName& OutAnchorKind)
{
    OutAnchorKind = FName(TEXT("nucleus"));
    if (!Plan.Events.IsValidIndex(AnchorEventIndex)) return NucleusCenterSec;
    EOffgridAIAudioLandmarkType ExpectedType;
    float ExpectedLeadSec = 0.0f;
    float VisualLeadSec = 0.0f;
    if (!VisualLandmarkForPhone(
            Plan.Events[AnchorEventIndex].SourcePhoneBase.ToUpper(),
            ExpectedType,
            ExpectedLeadSec,
            VisualLeadSec))
        return NucleusCenterSec;

    const float ExpectedCenterSec = NucleusCenterSec - ExpectedLeadSec;
    float BestScore = -1.0f;
    float BestCenterSec = NucleusCenterSec;
    for (const auto& Observation : Evidence)
    {
        if (Observation.Type != ExpectedType
            || Observation.DecisionSec > ObservedAudioBufferEndSec + 0.001f
            || Observation.CenterSec < NucleusCenterSec - 0.240f
            || Observation.CenterSec > NucleusCenterSec + 0.030f
            || Observation.CenterSec
                <= PreviousNucleusCenterSec + 0.015f
            || RegionContaining(SpeechRegions, Observation.CenterSec)
                != AudioRegionIndex)
            continue;
        const float Proximity = FMath::Clamp(
            1.0f - FMath::Abs(Observation.CenterSec - ExpectedCenterSec)
                / 0.180f,
            0.0f,
            1.0f);
        const float Score = Observation.Score * 0.70f + Proximity * 0.30f;
        if (Score <= BestScore) continue;
        BestScore = Score;
        BestCenterSec = Observation.CenterSec - VisualLeadSec;
    }
    if (BestScore < 0.50f) return NucleusCenterSec;
    switch (ExpectedType)
    {
    case EOffgridAIAudioLandmarkType::Bilabial:
        OutAnchorKind = FName(TEXT("bilabial_closure"));
        break;
    case EOffgridAIAudioLandmarkType::Labiodental:
        OutAnchorKind = FName(TEXT("labiodental_onset"));
        break;
    case EOffgridAIAudioLandmarkType::Sibilant:
        OutAnchorKind = FName(TEXT("sibilant_onset"));
        break;
    case EOffgridAIAudioLandmarkType::Glide:
        OutAnchorKind = FName(TEXT("glide_transition"));
        break;
    default:
        break;
    }
    return BestCenterSec;
}

static bool IsPerceptuallyImportantPhone(const FString& SourcePhoneBase)
{
    const FString Phone = SourcePhoneBase.ToUpper();
    if (Phone.IsEmpty() || Phone == TEXT("UNK")) return true;
    if (Phone == TEXT("AA") || Phone == TEXT("AE") || Phone == TEXT("AH")
        || Phone == TEXT("AO") || Phone == TEXT("AW") || Phone == TEXT("AY")
        || Phone == TEXT("EH") || Phone == TEXT("ER") || Phone == TEXT("EY")
        || Phone == TEXT("IH") || Phone == TEXT("IY") || Phone == TEXT("OW")
        || Phone == TEXT("OY") || Phone == TEXT("UH") || Phone == TEXT("UW"))
        return true;
    return Phone == TEXT("P") || Phone == TEXT("B") || Phone == TEXT("M")
        || Phone == TEXT("F") || Phone == TEXT("V")
        || Phone == TEXT("TH") || Phone == TEXT("DH")
        || Phone == TEXT("W") || Phone == TEXT("R") || Phone == TEXT("L")
        || Phone == TEXT("SH") || Phone == TEXT("ZH")
        || Phone == TEXT("CH") || Phone == TEXT("JH");
}

static int32 FirstPerceptuallyRenderedEventForWord(
    const FOffgridAITextVisemePlan& Plan,
    int32 WordIndex)
{
    for (int32 EventIndex = 0; EventIndex < Plan.Events.Num(); ++EventIndex)
    {
        const auto& Event = Plan.Events[EventIndex];
        if (Event.WordIndex == WordIndex && Event.bIsRenderable
            && IsPerceptuallyImportantPhone(Event.SourcePhoneBase))
            return EventIndex;
    }
    for (int32 EventIndex = 0; EventIndex < Plan.Events.Num(); ++EventIndex)
    {
        if (Plan.Events[EventIndex].WordIndex == WordIndex)
            return EventIndex;
    }
    return INDEX_NONE;
}

struct FRuntimePrior
{
    TArray<float> PhoneStarts;
    TArray<float> PhoneCenters;
    TArray<float> PhoneEnds;
    TArray<float> EventCenters;
    float TotalSec = 0.0f;
};

// Punctuation contributes hints to the planner, but never inserts runtime time.
// This prior is a continuous duration chain; observed audio regions own pauses.
static FRuntimePrior BuildRuntimePrior(const FOffgridAITextVisemePlan& Plan)
{
    FRuntimePrior Prior;
    Prior.PhoneStarts.SetNum(Plan.ExpectedPhones.Num());
    Prior.PhoneCenters.SetNum(Plan.ExpectedPhones.Num());
    Prior.PhoneEnds.SetNum(Plan.ExpectedPhones.Num());

    float Cursor = 0.0f;
    for (int32 Index = 0; Index < Plan.ExpectedPhones.Num(); ++Index)
    {
        const auto& Phone = Plan.ExpectedPhones[Index];
        if (Index > 0 && Plan.ExpectedPhones[Index - 1].WordIndex != Phone.WordIndex)
            Cursor += InterWordSeconds;
        const float Duration = FMath::Max(Phone.WeightSeconds, 0.025f);
        Prior.PhoneStarts[Index] = Cursor;
        Prior.PhoneCenters[Index] = Cursor + Duration * 0.5f;
        Prior.PhoneEnds[Index] = Cursor + Duration;
        Cursor += Duration;
    }
    Prior.TotalSec = Cursor;

    Prior.EventCenters.SetNum(Plan.Events.Num());
    for (int32 EventIndex = 0; EventIndex < Plan.Events.Num(); ++EventIndex)
    {
        const auto& Event = Plan.Events[EventIndex];
        if (Prior.PhoneCenters.IsValidIndex(Event.SourcePhoneGlobalIndex))
            Prior.EventCenters[EventIndex] = Prior.PhoneCenters[Event.SourcePhoneGlobalIndex];
        else
            Prior.EventCenters[EventIndex] = Plan.Events.Num() > 0
                ? Prior.TotalSec * (static_cast<float>(EventIndex) + 0.5f)
                    / static_cast<float>(Plan.Events.Num())
                : 0.0f;
    }
    return Prior;
}

static float LastCommittedCenter(const FOffgridAICommittedVisemeTrack& Track)
{
    return Track.Events.Num() > 0 ? Track.Events.Last().FinalRenderCenterSeconds : -1.0f;
}

static int32 CancelUnplayedPriorWordTail(
    FOffgridAICommittedVisemeTrack& Track,
    int32 NewWordIndex,
    float NewWordFirstCenterSec,
    float CurrentPlaybackSec)
{
    if (Track.Events.Num() == 0
        || Track.Events.Last().WordIndex == NewWordIndex
        || Track.Events.Last().FinalRenderCenterSeconds
            < NewWordFirstCenterSec - 0.0001f)
        return 0;

    const int32 PriorWordIndex = Track.Events.Last().WordIndex;
    int32 TailBegin = Track.Events.Num();
    for (int32 Index = Track.Events.Num() - 1; Index >= 0; --Index)
    {
        const auto& Event = Track.Events[Index];
        if (Event.WordIndex != PriorWordIndex) break;
        if (Event.FinalRenderCenterSeconds >= NewWordFirstCenterSec - 0.0001f
            && Event.FinalRenderCenterSeconds > CurrentPlaybackSec + 0.001f)
            TailBegin = Index;
    }
    if (TailBegin >= Track.Events.Num()) return 0;

    const float LowerBound = TailBegin > 0
        ? FMath::Max(
            Track.Events[TailBegin - 1].FinalRenderCenterSeconds + 0.00001f,
            CurrentPlaybackSec + 0.00001f)
        : CurrentPlaybackSec + 0.00001f;
    const float UpperBound = NewWordFirstCenterSec - 0.00001f;
    if (UpperBound <= LowerBound) return 0;

    const int32 Count = Track.Events.Num() - TailBegin;
    bool bPriorWordAlreadyHasVisibleEvent = false;
    for (int32 Index = 0; Index < TailBegin; ++Index)
    {
        const auto& Event = Track.Events[Index];
        if (Event.WordIndex == PriorWordIndex && Event.bIsRenderable
            && !Event.bCanceledByWordHandoff
            && IsPerceptuallyImportantPhone(Event.SourcePhoneBase))
        {
            bPriorWordAlreadyHasVisibleEvent = true;
            break;
        }
    }
    int32 PreservedVisibleEventIndex = INDEX_NONE;
    if (!bPriorWordAlreadyHasVisibleEvent)
    {
        for (int32 Index = TailBegin; Index < Track.Events.Num(); ++Index)
        {
            const auto& Event = Track.Events[Index];
            if (Event.bIsRenderable
                && IsPerceptuallyImportantPhone(Event.SourcePhoneBase))
            {
                PreservedVisibleEventIndex = Index;
                break;
            }
        }
    }
    const float Step = (UpperBound - LowerBound)
        / static_cast<float>(FMath::Max(Count, 1));
    int32 CanceledCount = 0;
    for (int32 Offset = 0; Offset < Count; ++Offset)
    {
        auto& Event = Track.Events[TailBegin + Offset];
        const float CenterSec = LowerBound
            + Step * static_cast<float>(Offset + 1);
        Event.FinalRenderCenterSeconds = CenterSec;
        const bool bPreserveVisibleNucleus =
            TailBegin + Offset == PreservedVisibleEventIndex;
        Event.bCanceledByWordHandoff = !bPreserveVisibleNucleus;
        if (!bPreserveVisibleNucleus)
        {
            Event.RenderStartSeconds = CenterSec;
            Event.RenderEndSeconds = CenterSec;
            Event.CommitReason = FName(TEXT("canceled_by_next_word_start"));
            ++CanceledCount;
        }
        else
        {
            Event.RenderStartSeconds = FMath::Min(
                Event.RenderStartSeconds, CenterSec);
            Event.RenderEndSeconds = NewWordFirstCenterSec;
            Event.CommitReason = FName(TEXT("preserved_word_nucleus_before_handoff"));
        }
        Event.AcousticAnchorKind = FName(TEXT("hard_word_handoff"));
        Event.AcousticAnchorSeconds = NewWordFirstCenterSec;
        Event.AcousticAnchorErrorSeconds = CenterSec - NewWordFirstCenterSec;
    }
    return CanceledCount;
}

static float CausalRegionEnd(const FOffgridAIStreamingSpeechRegion& Region);

static int32 FitUnplayedEventsToClosedRegions(
    FOffgridAICommittedVisemeTrack& Track,
    const TArray<FOffgridAIStreamingSpeechRegion>& Regions,
    float CurrentPlaybackSec)
{
    int32 FittedEventCount = 0;
    for (int32 RegionIndex = 0; RegionIndex < Regions.Num(); ++RegionIndex)
    {
        const auto& Region = Regions[RegionIndex];
        if (!Region.bEnded) continue;
        const float RegionEndSec = CausalRegionEnd(Region) - 0.020f;
        int32 TailWordIndex = INDEX_NONE;
        for (const auto& Event : Track.Events)
        {
            if (Event.SpeechRegionIndex == RegionIndex)
                TailWordIndex = Event.WordIndex;
        }
        int32 FirstFutureIndex = INDEX_NONE;
        int32 LastFutureIndex = INDEX_NONE;
        for (int32 EventIndex = 0; EventIndex < Track.Events.Num(); ++EventIndex)
        {
            const auto& Event = Track.Events[EventIndex];
            if (Event.SpeechRegionIndex != RegionIndex
                || Event.WordIndex != TailWordIndex
                || Event.FinalRenderCenterSeconds
                    <= CurrentPlaybackSec + 0.001f)
                continue;
            if (FirstFutureIndex == INDEX_NONE) FirstFutureIndex = EventIndex;
            LastFutureIndex = EventIndex;
        }
        if (FirstFutureIndex == INDEX_NONE
            || !Track.Events.IsValidIndex(LastFutureIndex))
            continue;

        // Preserve the word's perceptual start exactly. Only its still-unplayed
        // tail is elastic when a confirmed region close brackets the word.
        for (int32 EventIndex = FirstFutureIndex;
            EventIndex <= LastFutureIndex;
            ++EventIndex)
        {
            const auto& Event = Track.Events[EventIndex];
            if (Event.bIsRenderable && !Event.bCanceledByWordHandoff
                && IsPerceptuallyImportantPhone(Event.SourcePhoneBase))
            {
                FirstFutureIndex = EventIndex;
                break;
            }
        }

        const float OriginalFirstSec =
            Track.Events[FirstFutureIndex].FinalRenderCenterSeconds;
        const float OriginalLastSec =
            Track.Events[LastFutureIndex].FinalRenderCenterSeconds;
        if (OriginalLastSec <= RegionEndSec + 0.001f)
            continue;

        // A projected terminal word can land wholly beyond the closed audio
        // region even though it was committed early enough to be retimed. In
        // that one case preserve the latest real audio-confirmed word nucleus
        // and compress the complete unplayed suffix from that anchor. This
        // keeps every direct word start immutable while making the unmatched
        // terminal word visible before playback ends.
        bool bFitTerminalRecoverySuffix = false;
        if (OriginalFirstSec >= RegionEndSec - 0.001f)
        {
            for (int32 EventIndex = FirstFutureIndex - 1;
                EventIndex >= 0;
                --EventIndex)
            {
                const auto& Event = Track.Events[EventIndex];
                if (Event.SpeechRegionIndex != RegionIndex) break;
                if (Event.FinalRenderCenterSeconds
                        <= CurrentPlaybackSec + 0.001f
                    || Event.FinalRenderCenterSeconds
                        >= RegionEndSec - 0.001f
                    || Event.AcousticAnchorKind
                        != FName(TEXT("audio_word_start_nucleus"))
                    || FMath::Abs(Event.AcousticAnchorErrorSeconds) > 0.001f)
                    continue;
                FirstFutureIndex = EventIndex;
                bFitTerminalRecoverySuffix = true;
                break;
            }
            // In a low-preroll host the confirmed nucleus may already have
            // played by stream close. Preserve the latest still-unplayed event
            // from that confirmed word instead; committed history remains
            // immutable and only the future suffix is compressed.
            if (!bFitTerminalRecoverySuffix)
            {
                for (int32 EventIndex = FirstFutureIndex - 1;
                    EventIndex >= 0;
                    --EventIndex)
                {
                    const auto& Event = Track.Events[EventIndex];
                    if (Event.SpeechRegionIndex != RegionIndex) break;
                    if (Event.FinalRenderCenterSeconds
                            <= CurrentPlaybackSec + 0.001f
                        || Event.FinalRenderCenterSeconds
                            >= RegionEndSec - 0.001f)
                        continue;
                    FirstFutureIndex = EventIndex;
                    bFitTerminalRecoverySuffix = true;
                    break;
                }
            }
        }

        const float FitFirstSec =
            Track.Events[FirstFutureIndex].FinalRenderCenterSeconds;
        if (OriginalLastSec <= FitFirstSec + 0.001f
            || FitFirstSec >= RegionEndSec - 0.001f)
            continue;

        const float Scale = FMath::Clamp(
            (RegionEndSec - FitFirstSec)
                / (OriginalLastSec - FitFirstSec),
            0.05f,
            1.0f);
        for (int32 EventIndex = FirstFutureIndex;
            EventIndex <= LastFutureIndex;
            ++EventIndex)
        {
            auto& Event = Track.Events[EventIndex];
            if (Event.SpeechRegionIndex != RegionIndex
                || (!bFitTerminalRecoverySuffix
                    && Event.WordIndex != TailWordIndex)
                || Event.FinalRenderCenterSeconds
                    <= CurrentPlaybackSec + 0.001f)
                continue;
            const float OldCenterSec = Event.FinalRenderCenterSeconds;
            const float NewCenterSec = FitFirstSec
                + (OldCenterSec - FitFirstSec) * Scale;
            Event.RenderStartSeconds = NewCenterSec
                + (Event.RenderStartSeconds - OldCenterSec) * Scale;
            Event.RenderEndSeconds = NewCenterSec
                + (Event.RenderEndSeconds - OldCenterSec) * Scale;
            Event.FinalRenderCenterSeconds = NewCenterSec;
            Event.BoundaryOutcome = bFitTerminalRecoverySuffix
                ? FName(TEXT("terminal_recovery_suffix_fitted"))
                : FName(TEXT("region_tail_fitted"));
            ++FittedEventCount;
        }

        if (bFitTerminalRecoverySuffix)
        {
            float BoundedTerminalAnchorSec = -1.0f;
            for (int32 EventIndex = FirstFutureIndex;
                EventIndex <= LastFutureIndex;
                ++EventIndex)
            {
                const auto& Event = Track.Events[EventIndex];
                if (Event.WordIndex == TailWordIndex
                    && Event.AcousticAnchorKind
                        == FName(TEXT("projected_word_start"))
                    && FMath::Abs(Event.AcousticAnchorErrorSeconds) <= 0.001f)
                {
                    BoundedTerminalAnchorSec =
                        Event.FinalRenderCenterSeconds;
                    break;
                }
            }
            if (BoundedTerminalAnchorSec >= 0.0f)
            {
                for (int32 EventIndex = FirstFutureIndex;
                    EventIndex <= LastFutureIndex;
                    ++EventIndex)
                {
                    auto& Event = Track.Events[EventIndex];
                    if (Event.WordIndex != TailWordIndex) continue;
                    Event.AcousticAnchorKind =
                        FName(TEXT("bounded_terminal_word_recovery"));
                    Event.AcousticAnchorSeconds = BoundedTerminalAnchorSec;
                    Event.AcousticAnchorErrorSeconds =
                        Event.FinalRenderCenterSeconds
                        - BoundedTerminalAnchorSec;
                }
            }
        }
    }
    return FittedEventCount;
}

static float CausalRegionEnd(const FOffgridAIStreamingSpeechRegion& Region)
{
    return Region.bEnded && Region.ProvisionalEndSec >= 0.0f
        ? Region.ProvisionalEndSec
        : Region.AudioBufferEndSec;
}

static int32 RegionContaining(
    const TArray<FOffgridAIStreamingSpeechRegion>& Regions,
    float TimeSec)
{
    for (int32 Index = 0; Index < Regions.Num(); ++Index)
    {
        const float End = Regions[Index].bEnded
            ? CausalRegionEnd(Regions[Index])
            : Regions[Index].AudioBufferLastSpeechSec;
        if (TimeSec >= Regions[Index].AudioBufferStartSec && TimeSec <= End + 0.001f)
            return Index;
    }
    return INDEX_NONE;
}

static void FillCommittedEvent(
    const FOffgridAILipsyncRuntimeUpdateInput& Input,
    const FOffgridAITextVisemePlan& Plan,
    const FRuntimePrior& Prior,
    int32 EventIndex,
    int32 RegionIndex,
    float CenterSec,
    FName CommitReason,
    FOffgridAICommittedVisemeEvent& Out)
{
    const auto& Source = Plan.Events[EventIndex];
    const float Span = SpanForPose(Source.PoseID);
    Out.EventIndex = EventIndex;
    Out.PoseID = Source.PoseID;
    Out.Strength = Source.Strength;
    Out.JawOpenTarget = Source.JawOpenTarget;
    Out.SourceWord = Source.SourceText;
    Out.WordIndex = Source.WordIndex;
    Out.SentenceIndex = Source.SentenceIndex;
    Out.SpeechRegionIndex = RegionIndex;
    Out.bIsStrongVisibleEvent = Source.bIsStrongVisibleEvent;
    Out.bIsRenderable = Source.bIsRenderable;
    Out.TextCenterNorm = Prior.TotalSec > 0.0f
        ? Prior.EventCenters[EventIndex] / Prior.TotalSec
        : 0.0f;
    Out.TextDiagnosticCenterSeconds = Prior.EventCenters[EventIndex];
    Out.PriorCenterSeconds = Prior.EventCenters[EventIndex];
    Out.PriorStartSeconds = Prior.EventCenters[EventIndex] - Span * 0.5f;
    Out.PriorEndSeconds = Prior.EventCenters[EventIndex] + Span * 0.5f;
    Out.LeadAdjustedCenterSeconds = CenterSec;
    Out.FinalRenderCenterSeconds = CenterSec;
    Out.RenderStartSeconds = FMath::Max(CenterSec - Span * 0.5f, 0.0f);
    Out.RenderEndSeconds = CenterSec + Span * 0.5f;
    Out.SourcePhoneIndex = Source.SourcePhoneGlobalIndex;
    Out.SourcePhoneBase = Source.SourcePhoneBase;
    Out.SourcePhoneClass = PhoneClass(Source.SourcePhoneBase);
    Out.bMappedToObservedSpeech = RegionIndex != INDEX_NONE;
    Out.CommitPlaybackSeconds = Input.CurrentPlaybackSec;
    Out.CommitLeadSeconds = CenterSec - Input.CurrentPlaybackSec;
    Out.CommitReason = CommitReason;
    Out.bUsedInitialSpeechAnchor = RegionIndex == 0;
    Out.bUsedResumeAnchor = RegionIndex > 0;
    Out.AcousticAnchorKind = FName(TEXT("pending_word_anchor"));
    Out.AcousticAnchorSeconds = CenterSec;
    Out.AcousticAnchorErrorSeconds = CenterSec - Out.AcousticAnchorSeconds;
}
}

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
    PlaybackState = FOffgridAILipsyncSchedulerState();
}

void FOffgridAILipsyncRuntimeSession::BeginLine(const FOffgridAILipsyncRuntimeBeginInput& Input)
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
}

void FOffgridAILipsyncRuntimeSession::PushAudioPCM16(
    const TArray<uint8>& PCMChunk,
    int32 BytesToUse,
    int32 SampleRate,
    int32 NumChannels,
    int64 ChunkStartSample)
{
    if (!bBegun) return;
    Detector.AppendPCM16(PCMChunk, BytesToUse, SampleRate, NumChannels, ChunkStartSample);
    RefreshResolvedSpeechRegions();
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
}

void FOffgridAILipsyncRuntimeSession::CloseInputStream()
{
    bInputStreamClosed = true;
    Detector.Finalize(Detector.GetObservedAudioBufferEndSec());
    RefreshResolvedSpeechRegions();
}

void FOffgridAILipsyncRuntimeSession::Update(float CurrentPlaybackSec)
{
    if (!bBegun) return;
    PlaybackSec = FMath::Max(CurrentPlaybackSec, 0.0f);
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
    UpdateSyllablePacedVisemeTrack(
        Input, CommittedTrack, PlaybackState, bCommittedTrackBuilt);
    RecordRuntimeDiagnostics(PlaybackSec, false);
}

void FOffgridAILipsyncRuntimeSession::Finalize(float FinalPlaybackSec)
{
    if (!bBegun) return;
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
    UpdateSyllablePacedVisemeTrack(
        Input, CommittedTrack, PlaybackState, bCommittedTrackBuilt);
    RecordRuntimeDiagnostics(PlaybackSec, true);
}

void FOffgridAILipsyncRuntimeSession::RefreshResolvedSpeechRegions()
{
    ResolvedSpeechRegions = Detector.GetRefinedSpeechRegions();
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
    StreamTailDiagnosticRow.ObservedAudioBufferEndSec = Detector.GetObservedAudioBufferEndSec();
    StreamTailDiagnosticRow.FirstSpeechAudioBufferStartSec =
        Detector.HasObservedFirstSpeechStart() ? Detector.GetFirstSpeechAudioBufferStartSec() : 0.0f;
    StreamTailDiagnosticRow.SpeechRegionCount = ResolvedSpeechRegions.Num();
    StreamTailDiagnosticRow.bInputStreamClosed = bInputStreamClosed;
    StreamTailDiagnosticRow.DiagnosticKind = FName(TEXT("stream_tail"));

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
        Row.bContainsPlaybackSec = RegionContaining(ResolvedSpeechRegions, CurrentPlaybackSec)
            == Region.SpeechRegionIndex;
        Row.CloseReason = Region.EndReason;
        Row.CommittedEventCount = CommittedTrack.Events.Num();
        Row.DiagnosticKind = FName(TEXT("speech_region"));
        RuntimeSpeechRegionDiagnosticRows.Add(Row);
    }

    FOffgridAIRuntimeBoundaryDiagnosticRow StateRow;
    StateRow.LineID = LineID;
    StateRow.UpdateOrdinal = DiagnosticUpdateOrdinal;
    StateRow.bFinalReplay = bFinalReplay;
    StateRow.CurrentPlaybackSec = CurrentPlaybackSec;
    StateRow.bPlayheadStarted = PlaybackState.bPlayheadStarted;
    StateRow.ActiveSpeechRegionIndex = PlaybackState.ActiveSpeechRegionIndex;
    StateRow.ActiveTextSpeechRegionIndex = PlaybackState.ActiveTextSpeechRegionIndex;
    if (ResolvedSpeechRegions.IsValidIndex(PlaybackState.ActiveSpeechRegionIndex))
    {
        const auto& Region = ResolvedSpeechRegions[PlaybackState.ActiveSpeechRegionIndex];
        StateRow.ActiveRegionStartSec = Region.AudioBufferStartSec;
        StateRow.ActiveRegionEndSec = CausalRegionEnd(Region);
        StateRow.bAudioSpeechActive = CurrentPlaybackSec >= Region.AudioBufferStartSec
            && (!Region.bEnded || CurrentPlaybackSec <= CausalRegionEnd(Region));
    }
    StateRow.TimelineRate = PlaybackState.TimelineRate;
    StateRow.LastMatchedSyllableIndex = PlaybackState.LastMatchedSyllableIndex;
    StateRow.LastMatchedSyllablePhoneIndex = PlaybackState.LastMatchedSyllablePhoneIndex;
    StateRow.LastMatchedSyllableAudioSec = PlaybackState.LastMatchedSyllableAudioSec;
    StateRow.LastMatchedSyllableConfidence = PlaybackState.LastMatchedSyllableConfidence;
    StateRow.SchedulerNextEventIndex = PlaybackState.SchedulerNextEventIndex;
    StateRow.SchedulerNextPhoneIndex = PlaybackState.SchedulerNextPhoneIndex;
    StateRow.SchedulerCandidateCenterSec = PlaybackState.SchedulerCandidateCenterSec;
    StateRow.SchedulerCommitFrontierSec = PlaybackState.SchedulerCommitFrontierSec;
    StateRow.SchedulerCommitLeadSec = PlaybackState.SchedulerCommitLeadSec;
    StateRow.SchedulerBlockReason = PlaybackState.SchedulerBlockReason.ToString();
    StateRow.CommittedEventCount = CommittedTrack.Events.Num();
    StateRow.DiagnosticKind = FName(TEXT("audio_primary_scheduler"));
    RuntimeBoundaryDiagnosticRows.Add(StateRow);

    for (auto Row : PlaybackState.PendingSyllableAssignments)
    {
        Row.UpdateOrdinal = DiagnosticUpdateOrdinal;
        RuntimeSyllableAssignmentDiagnosticRows.Add(Row);
    }
    PlaybackState.PendingSyllableAssignments.Reset();

}

static void UpdateSyllablePacedVisemeTrack(
    const FOffgridAILipsyncRuntimeUpdateInput& Input,
    FOffgridAICommittedVisemeTrack& InOutTrack,
    FOffgridAILipsyncSchedulerState& State,
    bool& bInOutTrackBuilt)
{
    if (!Input.TextPlan || !Input.AudioFeatureFrames || !Input.SpeechRegions) return;
    const FOffgridAITextVisemePlan& Plan = *Input.TextPlan;
    const FRuntimePrior Prior = BuildRuntimePrior(Plan);

    InOutTrack.NPCID = Input.NPCID;
    InOutTrack.LineID = Input.LineID;
    InOutTrack.SpeechRegions.Reset();
    for (const auto& Region : *Input.SpeechRegions)
    {
        FOffgridAICommittedVisemeTrack::FSpeechRegion TrackRegion;
        TrackRegion.SpeechRegionIndex = Region.SpeechRegionIndex;
        TrackRegion.StartSeconds = Region.AudioBufferStartSec;
        TrackRegion.EndSeconds = Region.bEnded
            ? FMath::Max(Region.AudioBufferStartSec, CausalRegionEnd(Region) - 0.020f)
            : CausalRegionEnd(Region);
        TrackRegion.bEnded = Region.bEnded;
        InOutTrack.SpeechRegions.Add(TrackRegion);
    }
    FitUnplayedEventsToClosedRegions(
        InOutTrack,
        *Input.SpeechRegions,
        Input.CurrentPlaybackSec);

    FOffgridAIStreamingEvidenceSurfaceConfig Config;
    Config.PrerollSec = FMath::Max(Input.PrerollSec, 0.100f);
    Config.PostrollSec = SchedulerEvidenceHistorySec;
    Config.SpeechRegions = Input.SpeechRegions;
    const TArray<FOffgridAIAudioLandmarkObservation> Evidence =
        FOffgridAIStreamingEvidenceSurface::Analyze(*Input.AudioFeatureFrames, Config);
    const TArray<FOffgridAIStreamingSyllableCandidateSet> CandidateSets =
        FOffgridAIStreamingSyllablePositionEstimator::EstimateCandidateSets(
            Plan, Evidence, 4, 6, 4);

    float LastCenter = LastCommittedCenter(InOutTrack);
    for (int32 CandidateSetIndex = 0;
        CandidateSetIndex < CandidateSets.Num();
        ++CandidateSetIndex)
    {
        const auto& CandidateSet = CandidateSets[CandidateSetIndex];
        if (CandidateSet.AudioCenterSec <= State.LastProcessedSyllablePulseSec + 0.001f
            || (!Input.bInputStreamClosed
                && CandidateSet.DecisionSec + PulseCommitStabilitySec
                    > Input.ObservedAudioBufferEndSec + 0.001f))
            continue;

        const int32 AudioRegionIndex = RegionContaining(
            *Input.SpeechRegions, CandidateSet.AudioCenterSec);
        // A pulse can stabilize before the speech detector has published its
        // containing region. Leave it reconsiderable until region ownership
        // is known; committed correspondence is immutable after that point.
        if (AudioRegionIndex == INDEX_NONE) continue;
        int32 AssignedSyllableIndex = INDEX_NONE;
        float AssignmentScore = 0.0f;

        // At a transcript boundary, prefer an acoustically plausible
        // unresolved syllable of the active word over launching the following
        // word. This prevents a boundary-final nucleus (for example the second
        // vowel in "turkey," or "veggie.") from being relabeled as the next
        // word head. Punctuation does not create a pause or choose its timing;
        // it only breaks an otherwise ambiguous transcript correspondence.
        int32 ActiveWordContinuationSyllableIndex = INDEX_NONE;
        if (State.LastCommittedWordIndex >= 0
            && State.LastWordAnchorAudioRegionIndex == AudioRegionIndex)
        {
            const bool bActiveWordEndsTextRegion =
                Plan.WordBoundaryPunctuationAfter.IsValidIndex(
                    State.LastCommittedWordIndex)
                && Plan.WordBoundaryPunctuationAfter[
                    State.LastCommittedWordIndex] != TCHAR(0);
            if (bActiveWordEndsTextRegion)
            {
                for (int32 SyllableIndex = State.LastMatchedSyllableIndex + 1;
                    SyllableIndex < Plan.Syllables.Num();
                    ++SyllableIndex)
                {
                    const int32 WordIndex =
                        Plan.Syllables[SyllableIndex].WordIndex;
                    if (WordIndex == State.LastCommittedWordIndex)
                    {
                        ActiveWordContinuationSyllableIndex = SyllableIndex;
                        break;
                    }
                    if (WordIndex > State.LastCommittedWordIndex) break;
                }
            }
        }
        if (ActiveWordContinuationSyllableIndex != INDEX_NONE)
        {
            AssignedSyllableIndex = ActiveWordContinuationSyllableIndex;
            AssignmentScore = 0.45f;
            for (int32 CandidateIndex = 0;
                CandidateIndex < CandidateSet.SyllableIndices.Num();
                ++CandidateIndex)
            {
                if (CandidateSet.SyllableIndices[CandidateIndex]
                    != ActiveWordContinuationSyllableIndex)
                    continue;
                AssignedSyllableIndex = ActiveWordContinuationSyllableIndex;
                AssignmentScore = CandidateSet.Scores.IsValidIndex(CandidateIndex)
                    ? CandidateSet.Scores[CandidateIndex]
                    : 0.0f;
                break;
            }
        }
        for (int32 CandidateIndex = 0;
            AssignedSyllableIndex == INDEX_NONE
                && CandidateIndex < CandidateSet.SyllableIndices.Num();
            ++CandidateIndex)
        {
            const int32 SyllableIndex = CandidateSet.SyllableIndices[CandidateIndex];
            if (SyllableIndex <= State.LastMatchedSyllableIndex
                || !Plan.Syllables.IsValidIndex(SyllableIndex)
                || Plan.Syllables[SyllableIndex].WordIndex
                    > State.LastCommittedWordIndex + 1
                || Plan.Syllables[SyllableIndex].SpeechRegionIndex != AudioRegionIndex)
                continue;
            AssignedSyllableIndex = SyllableIndex;
            AssignmentScore = CandidateSet.Scores.IsValidIndex(CandidateIndex)
                ? CandidateSet.Scores[CandidateIndex]
                : 0.0f;
            break;
        }
        if (AssignedSyllableIndex == INDEX_NONE)
        {
            // Transcript punctuation and acoustic regions do not always have
            // the same ordinal topology. Continue the same monotonic
            // syllable alignment using the best forward candidate; observed
            // audio still owns the committed region, and word integrity is
            // checked below before anything becomes immutable.
            for (int32 CandidateIndex = 0;
                CandidateIndex < CandidateSet.SyllableIndices.Num();
                ++CandidateIndex)
            {
                const int32 SyllableIndex = CandidateSet.SyllableIndices[CandidateIndex];
                if (SyllableIndex <= State.LastMatchedSyllableIndex
                    || !Plan.Syllables.IsValidIndex(SyllableIndex)
                    || Plan.Syllables[SyllableIndex].WordIndex
                        > State.LastCommittedWordIndex + 1)
                    continue;
                AssignedSyllableIndex = SyllableIndex;
                AssignmentScore = CandidateSet.Scores.IsValidIndex(CandidateIndex)
                    ? CandidateSet.Scores[CandidateIndex]
                    : 0.0f;
                break;
            }
        }
        if (AssignedSyllableIndex == INDEX_NONE) continue;

        // A stable nucleus may advance at most one whole word. The beam may
        // skip syllables while resolving a missed or extra pulse, but it has
        // no authority to launch intervening words from a projected schedule.
        // Consequently every live word start is owned by its own immutable
        // accepted nucleus.
        int32 CandidateAssignedWordIndex =
            Plan.Syllables[AssignedSyllableIndex].WordIndex;
        if (CandidateAssignedWordIndex > State.LastCommittedWordIndex + 1)
        {
            for (int32 SyllableIndex = State.LastMatchedSyllableIndex + 1;
                SyllableIndex < Plan.Syllables.Num();
                ++SyllableIndex)
            {
                if (Plan.Syllables[SyllableIndex].WordIndex
                    == State.LastCommittedWordIndex + 1)
                {
                    AssignedSyllableIndex = SyllableIndex;
                    AssignmentScore = 0.45f;
                    break;
                }
            }
        }

        // Do not greedily launch a word from a later syllable.  Hold only
        // this ambiguous frontier long enough to observe the nearby pulse
        // budget, then move the correspondence back by the number of
        // supporting pulses actually present in the same audio region.  This
        // is a bounded monotonic alignment: it neither revises committed
        // words nor invents timing from the transcript.
        const int32 CandidateWordIndex =
            Plan.Syllables[AssignedSyllableIndex].WordIndex;
        const bool bBoundaryAfterWord =
            Plan.WordBoundaryPunctuationAfter.IsValidIndex(CandidateWordIndex)
            && Plan.WordBoundaryPunctuationAfter[CandidateWordIndex]
                != TCHAR(0);
        if (CandidateWordIndex == State.LastCommittedWordIndex + 1
            && bBoundaryAfterWord)
        {
            int32 FirstWordSyllableIndex = AssignedSyllableIndex;
            for (int32 SyllableIndex = AssignedSyllableIndex - 1;
                SyllableIndex >= 0;
                --SyllableIndex)
            {
                if (Plan.Syllables[SyllableIndex].WordIndex
                    != CandidateWordIndex)
                    break;
                FirstWordSyllableIndex = SyllableIndex;
            }
            if (AssignedSyllableIndex > FirstWordSyllableIndex)
            {
                const float FrontierReadySec = CandidateSet.AudioCenterSec
                    + AmbiguousWordHeadLookaheadSec;
                if (!Input.bInputStreamClosed
                    && Input.ObservedAudioBufferEndSec + 0.001f
                        < FrontierReadySec)
                    break;

                int32 SupportingFuturePulseCount = 0;
                for (int32 FutureSetIndex = CandidateSetIndex + 1;
                    FutureSetIndex < CandidateSets.Num();
                    ++FutureSetIndex)
                {
                    const auto& FutureSet = CandidateSets[FutureSetIndex];
                    if (FutureSet.AudioCenterSec
                        > CandidateSet.AudioCenterSec
                            + AmbiguousWordHeadLookaheadSec)
                        break;
                    if (RegionContaining(
                            *Input.SpeechRegions,
                            FutureSet.AudioCenterSec) == AudioRegionIndex)
                        ++SupportingFuturePulseCount;
                }
                AssignedSyllableIndex = FMath::Max(
                    FirstWordSyllableIndex,
                    AssignedSyllableIndex - SupportingFuturePulseCount);
            }
        }

        State.LastProcessedSyllablePulseSec = CandidateSet.AudioCenterSec;

        const FOffgridAIPlannedSyllable& AssignedSyllable =
            Plan.Syllables[AssignedSyllableIndex];

        // Additional nuclei inside an already committed multi-syllable word
        // refine correspondence, but can never release the following word.
        // Word ownership is atomic in this scheduler.
        if (AssignedSyllable.WordIndex <= State.LastCommittedWordIndex)
        {
            State.LastMatchedSyllableIndex = AssignedSyllableIndex;
            State.LastMatchedSyllablePhoneIndex = AssignedSyllable.NucleusPhoneIndex;
            State.LastMatchedSyllableAudioSec = CandidateSet.AudioCenterSec;
            State.LastMatchedSyllableConfidence =
                FMath::Clamp((AssignmentScore - 0.45f) / 1.20f, 0.0f, 1.0f);
            continue;
        }

        const int32 FirstWordIndex = State.LastCommittedWordIndex + 1;
        const int32 AssignedWordIndex = AssignedSyllable.WordIndex;
        const int32 AssignedAnchorEventIndex =
            FirstPerceptuallyRenderedEventForWord(Plan, AssignedWordIndex);
        if (!Prior.EventCenters.IsValidIndex(AssignedAnchorEventIndex)) continue;
        const float AssignedPriorAnchorSec = Prior.EventCenters[AssignedAnchorEventIndex];
        FName AssignedVisualAnchorKind = NAME_None;
        const float AssignedVisualAnchorSec =
            ResolveTranscriptConditionedVisualAnchor(
                Plan,
                AssignedAnchorEventIndex,
                Evidence,
                *Input.SpeechRegions,
                AudioRegionIndex,
                CandidateSet.AudioCenterSec,
                State.LastWordAnchorAudioSec,
                Input.ObservedAudioBufferEndSec,
                AssignedVisualAnchorKind);

        float ObservedWordIntervalSec = -1.0f;
        float PriorWordIntervalSec = -1.0f;
        if (FirstWordIndex == AssignedWordIndex
            && State.LastWordAnchorAudioRegionIndex == AudioRegionIndex
            && State.LastWordAnchorAudioSec >= 0.0f
            && State.LastWordAnchorPriorSec >= 0.0f)
        {
            ObservedWordIntervalSec = CandidateSet.AudioCenterSec
                - State.LastWordAnchorAudioSec;
            PriorWordIntervalSec = AssignedPriorAnchorSec
                - State.LastWordAnchorPriorSec;
            if (ObservedWordIntervalSec >= MinimumWordRateObservationSec
                && PriorWordIntervalSec > 0.0f)
            {
                const float ObservedRate = FMath::Clamp(
                    PriorWordIntervalSec / ObservedWordIntervalSec,
                    MinimumAdaptiveWordPriorRate,
                    MaximumAdaptiveWordPriorRate);
                State.AdaptiveWordPriorRate = FMath::Lerp(
                    State.AdaptiveWordPriorRate,
                    ObservedRate,
                    AdaptiveWordPriorRateBlend);
            }
        }
        const float FixedWordRate = FMath::Clamp(
            State.AdaptiveWordPriorRate,
            MinimumAdaptiveWordPriorRate,
            MaximumAdaptiveWordPriorRate);
        State.TimelineRate = FixedWordRate;

        const int32 RecoveredWordCount = AssignedWordIndex - FirstWordIndex;
        const int32 FirstRecoveryAnchorEventIndex =
            FirstPerceptuallyRenderedEventForWord(Plan, FirstWordIndex);
        const float FirstRecoveryPriorAnchorSec =
            Prior.EventCenters.IsValidIndex(FirstRecoveryAnchorEventIndex)
                ? Prior.EventCenters[FirstRecoveryAnchorEventIndex]
                : AssignedPriorAnchorSec;
        const float LiveRecoveryFloorSec =
            Input.CurrentPlaybackSec + MinLiveLeadSec;
        const float ProjectedFirstRecoveryAnchorSec = CandidateSet.AudioCenterSec
            - (AssignedPriorAnchorSec - FirstRecoveryPriorAnchorSec)
                / FixedWordRate;
        const bool bRedistributeLateRecovery = RecoveredWordCount > 0
            && ProjectedFirstRecoveryAnchorSec < LiveRecoveryFloorSec;
        auto RecoveryAnchorForWord = [&](int32 WordIndex, float PriorAnchorSec)
        {
            if (WordIndex == AssignedWordIndex)
                return AssignedVisualAnchorSec;
            if (!bRedistributeLateRecovery)
            {
                return CandidateSet.AudioCenterSec
                    - (AssignedPriorAnchorSec - PriorAnchorSec)
                        / FixedWordRate;
            }
            const float PriorSpan = FMath::Max(
                AssignedPriorAnchorSec - FirstRecoveryPriorAnchorSec,
                0.001f);
            const float Fraction = FMath::Clamp(
                (PriorAnchorSec - FirstRecoveryPriorAnchorSec) / PriorSpan,
                0.0f,
                1.0f);
            const float AvailableRecoverySpan = FMath::Max(
                CandidateSet.AudioCenterSec - LiveRecoveryFloorSec,
                0.001f * static_cast<float>(RecoveredWordCount));
            return FMath::Lerp(
                CandidateSet.AudioCenterSec - AvailableRecoverySpan,
                CandidateSet.AudioCenterSec,
                Fraction);
        };
        for (int32 WordIndex = FirstWordIndex;
            WordIndex <= AssignedWordIndex;
            ++WordIndex)
        {
            const int32 AnchorEventIndex =
                FirstPerceptuallyRenderedEventForWord(Plan, WordIndex);
            if (!Prior.EventCenters.IsValidIndex(AnchorEventIndex)) continue;
            const float PriorAnchorSec = Prior.EventCenters[AnchorEventIndex];
            const float WordAnchorSec = bRedistributeLateRecovery
                ? RecoveryAnchorForWord(WordIndex, PriorAnchorSec)
                : FMath::Max(
                    RecoveryAnchorForWord(WordIndex, PriorAnchorSec),
                    LiveRecoveryFloorSec);

            int32 FirstEventIndex = INDEX_NONE;
            int32 LastEventIndex = INDEX_NONE;
            for (int32 EventIndex = State.NextTextEventIndex;
                EventIndex < Plan.Events.Num();
                ++EventIndex)
            {
                if (Plan.Events[EventIndex].WordIndex != WordIndex)
                {
                    if (FirstEventIndex != INDEX_NONE) break;
                    continue;
                }
                if (FirstEventIndex == INDEX_NONE) FirstEventIndex = EventIndex;
                LastEventIndex = EventIndex;
            }
            if (FirstEventIndex == INDEX_NONE) continue;

            // When a later confirmed anchor brackets a recovered word, fit
            // that word's complete prior inside the available interval with
            // one uniform rate. Relative intra-word timing remains intact.
            float WordPriorRate = FixedWordRate;
            if (WordIndex < AssignedWordIndex)
            {
                const int32 NextWordIndex = WordIndex + 1;
                const int32 NextAnchorEventIndex =
                    FirstPerceptuallyRenderedEventForWord(Plan, NextWordIndex);
                int32 NextFirstEventIndex = INDEX_NONE;
                for (int32 EventIndex = LastEventIndex + 1;
                    EventIndex < Plan.Events.Num();
                    ++EventIndex)
                {
                    if (Plan.Events[EventIndex].WordIndex == NextWordIndex)
                    {
                        NextFirstEventIndex = EventIndex;
                        break;
                    }
                }
                if (Prior.EventCenters.IsValidIndex(NextAnchorEventIndex)
                    && Prior.EventCenters.IsValidIndex(NextFirstEventIndex))
                {
                    const float NextPriorAnchorSec =
                        Prior.EventCenters[NextAnchorEventIndex];
                    const float NextWordAnchorSec = bRedistributeLateRecovery
                        ? RecoveryAnchorForWord(
                            NextWordIndex, NextPriorAnchorSec)
                        : FMath::Max(
                            RecoveryAnchorForWord(
                                NextWordIndex, NextPriorAnchorSec),
                            LiveRecoveryFloorSec);
                    const float NextLeadingSpan = FMath::Max(
                        NextPriorAnchorSec
                            - Prior.EventCenters[NextFirstEventIndex],
                        0.0f) / FixedWordRate;
                    const float AvailableTailSec = NextWordAnchorSec
                        - NextLeadingSpan - WordAnchorSec - 0.001f;
                    const float PriorTailSec = FMath::Max(
                        Prior.EventCenters[LastEventIndex] - PriorAnchorSec,
                        0.0f);
                    if (AvailableTailSec > 0.001f)
                    {
                        WordPriorRate = FMath::Max(
                            WordPriorRate,
                            PriorTailSec / AvailableTailSec);
                    }
                }
            }

            // Keep the visually meaningful word start exactly on the audio
            // nucleus. Leading articulation uses its trained relative prior;
            // only that leading span may be proportionally compressed if the
            // causal live lead is smaller than the trained anticipation.
            const float PriorFirstCenter = Prior.EventCenters[FirstEventIndex];
            const float PriorLeadingSpan = FMath::Max(
                PriorAnchorSec - PriorFirstCenter, 0.0f);
            const float AvailableLeadingSpan = FMath::Max(
                WordAnchorSec - (Input.CurrentPlaybackSec + MinLiveLeadSec),
                0.0f);
            const float LeadingScale = PriorLeadingSpan > 0.0f
                ? FMath::Min(1.0f / WordPriorRate,
                    AvailableLeadingSpan / PriorLeadingSpan)
                : 1.0f / WordPriorRate;
            const float DesiredFirstCenterSec = FMath::Max(
                WordAnchorSec
                    + (PriorFirstCenter - PriorAnchorSec) * LeadingScale,
                Input.CurrentPlaybackSec + MinLiveLeadSec);
            const int32 LeadingTimingEventCount = FMath::Max(
                AnchorEventIndex - FirstEventIndex,
                0);
            const float HandoffTailCutoffSec = WordAnchorSec
                - 0.001f * static_cast<float>(LeadingTimingEventCount);
            // Only an independently audio-confirmed B may terminate A.
            // Projected recovery words have no authority to erase animation.
            const int32 CanceledPriorWordEventCount =
                WordIndex == AssignedWordIndex
                    ? CancelUnplayedPriorWordTail(
                        InOutTrack,
                        WordIndex,
                        HandoffTailCutoffSec,
                        Input.CurrentPlaybackSec)
                    : 0;
            if (CanceledPriorWordEventCount > 0)
                LastCenter = LastCommittedCenter(InOutTrack);

            for (int32 EventIndex = FirstEventIndex;
                EventIndex <= LastEventIndex;
                ++EventIndex)
            {
                const float PriorOffset = Prior.EventCenters[EventIndex]
                    - PriorAnchorSec;
                float CenterSec = WordAnchorSec + (PriorOffset < 0.0f
                    ? PriorOffset * LeadingScale
                    : PriorOffset / WordPriorRate);
                CenterSec = FMath::Max(
                    CenterSec, Input.CurrentPlaybackSec + MinLiveLeadSec);
                if (LastCenter >= 0.0f)
                    CenterSec = FMath::Max(CenterSec, LastCenter + 0.001f);

                FOffgridAICommittedVisemeEvent Event;
                FillCommittedEvent(
                    Input,
                    Plan,
                    Prior,
                    EventIndex,
                    AudioRegionIndex,
                    CenterSec,
                    FName(TEXT("audio_word_anchor_prior_commit")),
                    Event);
                Event.AcousticAnchorKind = FName(TEXT("audio_word_start_nucleus"));
                Event.AcousticAnchorSeconds = WordAnchorSec;
                Event.AcousticAnchorErrorSeconds = CenterSec - WordAnchorSec;
                InOutTrack.Events.Add(Event);
                LastCenter = CenterSec;
                State.NextTextEventIndex = EventIndex + 1;
            }

            FOffgridAIRuntimeSyllableAssignmentDiagnosticRow Row;
            Row.LineID = Input.LineID;
            Row.AudioSpeechRegionIndex = AudioRegionIndex;
            Row.TextSpeechRegionIndex = Plan.WordSpeechRegionIndices.IsValidIndex(WordIndex)
                ? Plan.WordSpeechRegionIndices[WordIndex]
                : AssignedSyllable.SpeechRegionIndex;
            Row.PhoneIndex = WordIndex == AssignedWordIndex
                ? AssignedSyllable.NucleusPhoneIndex
                : INDEX_NONE;
            Row.ObservedAudioSec = WordAnchorSec;
            Row.Prominence = WordIndex == AssignedWordIndex ? AssignmentScore : 0.0f;
            Row.Confidence = WordIndex == AssignedWordIndex
                ? FMath::Clamp((AssignmentScore - 0.45f) / 1.20f, 0.0f, 1.0f)
                : 0.0f;
            Row.SkipCount = RecoveredWordCount;
            Row.AnchorKind = WordIndex == AssignedWordIndex
                ? FName(TEXT("audio_word_start_prior"))
                : FName(TEXT("audio_word_start_prior_recovery"));
            Row.WordIndex = WordIndex;
            Row.WordPriorRate = WordPriorRate;
            Row.ObservedWordIntervalSec = ObservedWordIntervalSec;
            Row.PriorWordIntervalSec = PriorWordIntervalSec;
            Row.CanceledPriorWordEventCount = CanceledPriorWordEventCount;
            Row.NucleusAudioSec = WordIndex == AssignedWordIndex
                ? CandidateSet.AudioCenterSec
                : -1.0f;
            Row.VisualAnchorAudioSec = WordAnchorSec;
            Row.VisualAnchorKind = WordIndex == AssignedWordIndex
                ? AssignedVisualAnchorKind
                : FName(TEXT("recovered_prior"));
            Row.VisualAnchorPhoneIndex = WordIndex == AssignedWordIndex
                && Plan.Events.IsValidIndex(AssignedAnchorEventIndex)
                ? Plan.Events[AssignedAnchorEventIndex].SourcePhoneGlobalIndex
                : INDEX_NONE;
            State.PendingSyllableAssignments.Add(Row);
            State.LastCommittedWordIndex = WordIndex;
        }

        State.LastWordAnchorAudioRegionIndex = AudioRegionIndex;
        State.LastWordAnchorAudioSec = CandidateSet.AudioCenterSec;
        State.LastWordAnchorPriorSec = AssignedPriorAnchorSec;

        State.LastMatchedSyllableIndex = AssignedSyllableIndex;
        State.LastMatchedSyllablePhoneIndex = AssignedSyllable.NucleusPhoneIndex;
        State.LastMatchedSyllableAudioSec = CandidateSet.AudioCenterSec;
        State.LastMatchedSyllableConfidence =
            FMath::Clamp((AssignmentScore - 0.45f) / 1.20f, 0.0f, 1.0f);
        State.ActiveSpeechRegionIndex = AudioRegionIndex;
        State.ActiveTextSpeechRegionIndex = AssignedSyllable.SpeechRegionIndex;
        State.bPlayheadStarted = true;
    }

    // A nucleus can be missed at the stream tail (there is no later rebound
    // with which to stabilize it). Preserve transcript completeness by
    // projecting only the remaining whole words from the last observed word
    // anchor and the already learned rate. This is explicitly diagnosed as a
    // lower-confidence recovery, never mistaken for an acoustic word start.
    if (Input.bInputStreamClosed
        && State.LastCommittedWordIndex + 1 < Plan.WordPhoneBeginIndices.Num()
        && State.LastCommittedWordIndex + 2 == Plan.WordPhoneBeginIndices.Num()
        && State.LastWordAnchorAudioSec >= 0.0f)
    {
        const int32 AudioRegionIndex = Input.SpeechRegions->Num() > 0
            ? Input.SpeechRegions->Num() - 1
            : INDEX_NONE;
        const float FixedWordRate = FMath::Clamp(
            State.AdaptiveWordPriorRate,
            MinimumAdaptiveWordPriorRate,
            MaximumAdaptiveWordPriorRate);
        const bool bOnlyFinalWordRemains =
            State.LastCommittedWordIndex + 2
                == Plan.WordPhoneBeginIndices.Num();
        float TerminalNucleusRecoverySec = -1.0f;
        float TerminalNucleusRecoveryScore = -1.0f;
        if (bOnlyFinalWordRemains
            && Input.SpeechRegions->IsValidIndex(AudioRegionIndex))
        {
            const auto& FinalRegion = (*Input.SpeechRegions)[AudioRegionIndex];
            const float FinalRegionEndSec = CausalRegionEnd(FinalRegion);
            for (const auto& Observation : Evidence)
            {
                if (Observation.Type
                        != EOffgridAIAudioLandmarkType::SyllabicPulse
                    || Observation.CenterSec
                        <= State.LastMatchedSyllableAudioSec + 0.001f
                    || Observation.CenterSec
                        < FinalRegion.AudioBufferStartSec - 0.001f
                    || Observation.CenterSec > FinalRegionEndSec + 0.001f)
                    continue;
                if (Observation.Score > TerminalNucleusRecoveryScore)
                {
                    TerminalNucleusRecoverySec = Observation.CenterSec;
                    TerminalNucleusRecoveryScore = Observation.Score;
                }
            }
        }
        float LastTailRecoveryAnchorSec = -1.0f;
        for (int32 WordIndex = State.LastCommittedWordIndex + 1;
            WordIndex < Plan.WordPhoneBeginIndices.Num();
            ++WordIndex)
        {
            // Never manufacture a word start from elapsed transcript time.
            // The sole terminal recovery is legal only when an unmatched
            // observed nucleus exists in the final audio region.
            if (TerminalNucleusRecoverySec < 0.0f) break;
            const int32 AnchorEventIndex =
                FirstPerceptuallyRenderedEventForWord(Plan, WordIndex);
            if (!Prior.EventCenters.IsValidIndex(AnchorEventIndex)) continue;
            const float PriorAnchorSec = Prior.EventCenters[AnchorEventIndex];
            const float ProjectedAnchorSec = State.LastWordAnchorAudioSec
                + (PriorAnchorSec - State.LastWordAnchorPriorSec) / FixedWordRate;
            const bool bUseTerminalNucleusRecovery =
                bOnlyFinalWordRemains
                && WordIndex + 1 == Plan.WordPhoneBeginIndices.Num()
                && TerminalNucleusRecoverySec >= 0.0f;
            const float WordAnchorSec = FMath::Max(
                FMath::Max(
                    bUseTerminalNucleusRecovery
                        ? TerminalNucleusRecoverySec
                        : ProjectedAnchorSec,
                    Input.CurrentPlaybackSec + MinLiveLeadSec),
                LastTailRecoveryAnchorSec >= 0.0f
                    ? LastTailRecoveryAnchorSec + 0.001f
                    : -1.0f);
            LastTailRecoveryAnchorSec = WordAnchorSec;

            int32 FirstEventIndex = INDEX_NONE;
            int32 LastEventIndex = INDEX_NONE;
            for (int32 EventIndex = State.NextTextEventIndex;
                EventIndex < Plan.Events.Num();
                ++EventIndex)
            {
                if (Plan.Events[EventIndex].WordIndex != WordIndex)
                {
                    if (FirstEventIndex != INDEX_NONE) break;
                    continue;
                }
                if (FirstEventIndex == INDEX_NONE) FirstEventIndex = EventIndex;
                LastEventIndex = EventIndex;
            }
            if (FirstEventIndex == INDEX_NONE) continue;

            const float PriorFirstCenter = Prior.EventCenters[FirstEventIndex];
            const float PriorLeadingSpan = FMath::Max(
                PriorAnchorSec - PriorFirstCenter, 0.0f);
            const float AvailableLeadingSpan = FMath::Max(
                WordAnchorSec - (Input.CurrentPlaybackSec + MinLiveLeadSec),
                0.0f);
            const float LeadingScale = PriorLeadingSpan > 0.0f
                ? FMath::Min(1.0f / FixedWordRate,
                    AvailableLeadingSpan / PriorLeadingSpan)
                : 1.0f / FixedWordRate;
            for (int32 EventIndex = FirstEventIndex;
                EventIndex <= LastEventIndex;
                ++EventIndex)
            {
                const float PriorOffset = Prior.EventCenters[EventIndex]
                    - PriorAnchorSec;
                float CenterSec = WordAnchorSec + (PriorOffset < 0.0f
                    ? PriorOffset * LeadingScale
                    : PriorOffset / FixedWordRate);
                CenterSec = FMath::Max(
                    CenterSec, Input.CurrentPlaybackSec + MinLiveLeadSec);
                if (LastCenter >= 0.0f)
                    CenterSec = FMath::Max(CenterSec, LastCenter + 0.001f);
                FOffgridAICommittedVisemeEvent Event;
                FillCommittedEvent(
                    Input, Plan, Prior, EventIndex, AudioRegionIndex,
                    CenterSec,
                    FName(TEXT("audio_word_prior_stream_tail_recovery")),
                    Event);
                Event.AcousticAnchorKind = FName(TEXT("projected_word_start"));
                if (bUseTerminalNucleusRecovery)
                {
                    Event.AcousticAnchorKind =
                        FName(TEXT("terminal_unmatched_nucleus"));
                }
                Event.AcousticAnchorSeconds = WordAnchorSec;
                Event.AcousticAnchorErrorSeconds = CenterSec - WordAnchorSec;
                InOutTrack.Events.Add(Event);
                LastCenter = CenterSec;
                State.NextTextEventIndex = EventIndex + 1;
            }

            FOffgridAIRuntimeSyllableAssignmentDiagnosticRow Row;
            Row.LineID = Input.LineID;
            Row.AudioSpeechRegionIndex = AudioRegionIndex;
            Row.TextSpeechRegionIndex = Plan.WordSpeechRegionIndices.IsValidIndex(WordIndex)
                ? Plan.WordSpeechRegionIndices[WordIndex]
                : INDEX_NONE;
            Row.ObservedAudioSec = WordAnchorSec;
            Row.AnchorKind = bUseTerminalNucleusRecovery
                ? FName(TEXT("audio_word_start_terminal_nucleus_recovery"))
                : FName(TEXT("audio_word_start_prior_stream_tail_recovery"));
            Row.WordIndex = WordIndex;
            Row.WordPriorRate = FixedWordRate;
            State.PendingSyllableAssignments.Add(Row);
            State.LastCommittedWordIndex = WordIndex;
        }
    }

    // Include words committed during this final update. Region fitting is
    // idempotent and only touches centers that have not reached playback.
    FitUnplayedEventsToClosedRegions(
        InOutTrack,
        *Input.SpeechRegions,
        Input.CurrentPlaybackSec);

    // Region fitting may replace a projected terminal anchor with its bounded
    // in-region center. Keep the diagnostic row tied to the performed anchor.
    for (auto& Row : State.PendingSyllableAssignments)
    {
        if (Row.AnchorKind
            != FName(TEXT("audio_word_start_prior_stream_tail_recovery")))
            continue;
        const int32 AnchorEventIndex =
            FirstPerceptuallyRenderedEventForWord(Plan, Row.WordIndex);
        for (const auto& Event : InOutTrack.Events)
        {
            if (Event.EventIndex != AnchorEventIndex
                || Event.AcousticAnchorKind
                    != FName(TEXT("bounded_terminal_word_recovery")))
                continue;
            Row.ObservedAudioSec = Event.AcousticAnchorSeconds;
            Row.AnchorKind =
                FName(TEXT("audio_word_start_bounded_terminal_recovery"));
            break;
        }
    }

    State.SchedulerNextEventIndex = State.NextTextEventIndex;
    State.SchedulerNextPhoneIndex = Plan.Syllables.IsValidIndex(
        State.LastMatchedSyllableIndex + 1)
        ? Plan.Syllables[State.LastMatchedSyllableIndex + 1].NucleusPhoneIndex
        : INDEX_NONE;
    State.SchedulerCandidateCenterSec = State.LastMatchedSyllableAudioSec;
    State.SchedulerCommitFrontierSec = Input.ObservedAudioBufferEndSec;
    State.SchedulerCommitLeadSec = State.LastMatchedSyllableAudioSec
        - Input.CurrentPlaybackSec;
    State.SchedulerBlockReason = State.NextTextEventIndex >= Plan.Events.Num()
        ? FName(TEXT("complete"))
        : FName(TEXT("waiting_for_stable_word_start_correspondence"));
    if (Input.SpeechRegions->Num() > 0)
    {
        InOutTrack.SpeechStartSeconds = (*Input.SpeechRegions)[0].AudioBufferStartSec;
        InOutTrack.SpeechEndSeconds = Input.SpeechRegions->Last().AudioBufferEndSec;
    }
    bInOutTrackBuilt = true;
}
