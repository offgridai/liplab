#include "Lipsync/OffgridAILipsyncRuntimeAdapter.h"
#include "Lipsync/OffgridAILipsyncVersion.h"

#include "Lipsync/OffgridAIStreamingEvidenceSurface.h"
#include "Lipsync/OffgridAIStreamingSyllablePositionEstimator.h"

namespace
{
static constexpr float InterWordSeconds = 0.020f;
static constexpr float MaxCommitLeadSec = 0.160f;
static constexpr float StandardCommaMaxCommitLeadSec = 0.120f;
static constexpr float MinLiveLeadSec = 0.030f;
static constexpr float MinEventSpacingSec = 0.035f;
static constexpr float MinRegionTailCompactionSpacingSec = 0.001f;
static constexpr float MaxRegionTailCompactionOverrunSec = 0.020f;
static constexpr float MaxAtomicWordTailOverrunSec = 0.040f;
static constexpr float NominalPriorPlaybackRate = 0.90f;
static constexpr float FocusedPriorPlaybackRate = 1.05f;
static constexpr float FocusedMinimumAdaptiveRate = FocusedPriorPlaybackRate;
static constexpr float FocusedMaximumAdaptiveRate = 1.20f;
static constexpr float FocusedAdaptiveRateBlend = 0.50f;
static constexpr float FocusedMinimumRateSpanSec = 0.180f;
static constexpr float HardVirtualResumeMinimumQuietSec = 0.080f;
static constexpr float MinAnchorMatchScore = 0.75f;
static constexpr float MaxSyllableRebaseSec = 0.120f;
static constexpr float TextRegionMergeEvidenceSec = 0.300f;
// Direct rendering cannot retract a pulse if a stronger nearby peak appears
// later. Wait one de-duplication horizon beyond the evidence decision so each
// immutable open-mouth event represents the stable pulse candidate.
static constexpr float AudioPulseCommitStabilitySec = 0.120f;
static constexpr float MinimumAdaptiveWordPriorRate = 0.65f;
static constexpr float MaximumAdaptiveWordPriorRate = 1.65f;
static constexpr float AdaptiveWordPriorRateBlend = 0.35f;
static constexpr float MinimumWordRateObservationSec = 0.080f;
static constexpr float ListRestartMinimumValleySec = 0.020f;
static constexpr float ListRestartMaximumValleySec = 0.180f;
static constexpr float ListRestartMaximumValleyRMSNorm = 0.020f;
static constexpr float ListRestartMinimumReboundRatio = 3.0f;
static constexpr float ListRestartMinimumFlux = 0.060f;
static constexpr int32 RecognizedListMinimumBoundaryCount = 1;
static constexpr int32 DenseListMinimumBoundaryCount = 6;

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
        if (OriginalLastSec <= RegionEndSec + 0.001f
            || OriginalFirstSec >= RegionEndSec - 0.001f
            || OriginalLastSec <= OriginalFirstSec + 0.001f)
            continue;

        const float Scale = FMath::Clamp(
            (RegionEndSec - OriginalFirstSec)
                / (OriginalLastSec - OriginalFirstSec),
            0.05f,
            1.0f);
        for (int32 EventIndex = FirstFutureIndex;
            EventIndex <= LastFutureIndex;
            ++EventIndex)
        {
            auto& Event = Track.Events[EventIndex];
            if (Event.SpeechRegionIndex != RegionIndex
                || Event.WordIndex != TailWordIndex
                || Event.FinalRenderCenterSeconds
                    <= CurrentPlaybackSec + 0.001f)
                continue;
            const float OldCenterSec = Event.FinalRenderCenterSeconds;
            const float NewCenterSec = OriginalFirstSec
                + (OldCenterSec - OriginalFirstSec) * Scale;
            Event.RenderStartSeconds = NewCenterSec
                + (Event.RenderStartSeconds - OldCenterSec) * Scale;
            Event.RenderEndSeconds = NewCenterSec
                + (Event.RenderEndSeconds - OldCenterSec) * Scale;
            Event.FinalRenderCenterSeconds = NewCenterSec;
            Event.BoundaryOutcome = FName(TEXT("region_tail_fitted"));
            ++FittedEventCount;
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

static float PriorToAudio(const FOffgridAIBoundaryPlaybackState& State, float PriorSec)
{
    return State.TimelineAudioAnchorSec
        + (PriorSec - State.TimelinePriorAnchorSec) / FMath::Max(State.TimelineRate, 0.10f);
}

static bool IsStandardCommaBoundary(
    const FOffgridAITextVisemePlan& Plan,
    int32 BoundaryWordIndex)
{
    if (!Plan.WordBoundaryPunctuationAfter.IsValidIndex(BoundaryWordIndex)
        || Plan.WordBoundaryPunctuationAfter[BoundaryWordIndex] != TEXT(','))
        return false;

    for (int32 Index = BoundaryWordIndex + 1;
        Index < Plan.WordBoundaryPunctuationAfter.Num();
        ++Index)
    {
        if (Plan.WordBoundaryPunctuationAfter[Index] == TCHAR(0))
            continue;

        // Another comma one word later is a list comma. More than one word
        // before the next punctuation makes this a standard comma.
        return Index > BoundaryWordIndex + 1;
    }
    return false;
}

static bool IsListGapSensitiveAtCursor(
    const FOffgridAITextVisemePlan& Plan,
    const FOffgridAIBoundaryPlaybackState& State)
{
    if (Plan.Events.Num() == 0 || Plan.WordBoundaryPauseClassAfter.Num() == 0)
        return false;

    const int32 EventIndex = FMath::Clamp(State.NextTextEventIndex, 0, Plan.Events.Num() - 1);
    const int32 WordIndex = Plan.Events[EventIndex].WordIndex;
    if (WordIndex < 0)
        return false;

    int32 SentenceStart = 0;
    for (int32 Index = WordIndex - 1; Index >= 0; --Index)
    {
        if (Plan.WordBoundaryPauseClassAfter.IsValidIndex(Index)
            && Plan.WordBoundaryPauseClassAfter[Index]
                == EOffgridAIBoundaryPauseClass::HardBreakPause)
        {
            SentenceStart = Index + 1;
            break;
        }
    }

    int32 SentenceEnd = Plan.WordBoundaryPauseClassAfter.Num() - 1;
    for (int32 Index = WordIndex; Index < Plan.WordBoundaryPauseClassAfter.Num(); ++Index)
    {
        if (Plan.WordBoundaryPauseClassAfter[Index]
            == EOffgridAIBoundaryPauseClass::HardBreakPause)
        {
            SentenceEnd = Index;
            break;
        }
    }

    int32 FirstListBoundary = INDEX_NONE;
    for (int32 Index = SentenceStart; Index <= SentenceEnd; ++Index)
    {
        if (Plan.WordBoundaryPauseClassAfter[Index]
            == EOffgridAIBoundaryPauseClass::SoftListPause)
        {
            FirstListBoundary = Index;
            break;
        }
    }

    // Enter as the cursor reaches the first comma-delimited item and remain
    // sensitive through its sentence-final boundary. Audio still decides
    // whether any actual quiet run becomes a split.
    return FirstListBoundary != INDEX_NONE
        && WordIndex >= FirstListBoundary
        && WordIndex <= SentenceEnd;
}

static bool IsRecognizedListActive(
    const FOffgridAITextVisemePlan& Plan,
    const FOffgridAIBoundaryPlaybackState& State)
{
    if (Plan.Events.Num() == 0 || Plan.WordBoundaryPauseClassAfter.Num() == 0)
        return false;

    const int32 EventIndex = FMath::Clamp(State.NextTextEventIndex, 0, Plan.Events.Num() - 1);
    const int32 CursorWordIndex = Plan.Events[EventIndex].WordIndex;
    const int32 LatchedListWordIndex = State.PendingListBoundaryWordIndex != INDEX_NONE
        ? State.PendingListBoundaryWordIndex
        : State.LastResolvedListBoundaryWordIndex;

    int32 SentenceStart = 0;
    while (SentenceStart < Plan.WordBoundaryPauseClassAfter.Num())
    {
        int32 SentenceEnd = Plan.WordBoundaryPauseClassAfter.Num() - 1;
        int32 FirstListBoundary = INDEX_NONE;
        int32 SoftBoundaryCount = 0;
        for (int32 Index = SentenceStart;
            Index < Plan.WordBoundaryPauseClassAfter.Num();
            ++Index)
        {
            const EOffgridAIBoundaryPauseClass PauseClass =
                Plan.WordBoundaryPauseClassAfter[Index];
            if (PauseClass == EOffgridAIBoundaryPauseClass::SoftListPause)
            {
                if (FirstListBoundary == INDEX_NONE)
                    FirstListBoundary = Index;
                ++SoftBoundaryCount;
            }
            if (PauseClass == EOffgridAIBoundaryPauseClass::HardBreakPause)
            {
                SentenceEnd = Index;
                break;
            }
        }

        if (SoftBoundaryCount >= RecognizedListMinimumBoundaryCount
            && FirstListBoundary != INDEX_NONE)
        {
            if (CursorWordIndex >= FirstListBoundary && CursorWordIndex <= SentenceEnd)
                return true;

            // The text scheduler may reach the following sentence before all
            // of the list audio has arrived. Once a list boundary is pending
            // or resolved, retain sensitivity until its real hard break has
            // been acoustically resolved; do not leak it into later speech.
            if (LatchedListWordIndex >= FirstListBoundary
                && LatchedListWordIndex <= SentenceEnd + 1
                && State.LastResolvedPauseBoundaryWordIndex < SentenceEnd)
            {
                return true;
            }
        }

        SentenceStart = SentenceEnd + 1;
    }
    return false;
}

static int32 ListBoundaryCountInSentence(
    const FOffgridAITextVisemePlan& Plan,
    int32 WordIndex)
{
    if (WordIndex < 0 || Plan.WordBoundaryPauseClassAfter.Num() == 0)
        return 0;

    int32 SentenceStart = 0;
    for (int32 Index = WordIndex - 1; Index >= 0; --Index)
    {
        if (Plan.WordBoundaryPauseClassAfter.IsValidIndex(Index)
            && Plan.WordBoundaryPauseClassAfter[Index]
                == EOffgridAIBoundaryPauseClass::HardBreakPause)
        {
            SentenceStart = Index + 1;
            break;
        }
    }

    int32 SentenceEnd = Plan.WordBoundaryPauseClassAfter.Num() - 1;
    for (int32 Index = WordIndex; Index < Plan.WordBoundaryPauseClassAfter.Num(); ++Index)
    {
        if (Plan.WordBoundaryPauseClassAfter[Index]
            == EOffgridAIBoundaryPauseClass::HardBreakPause)
        {
            SentenceEnd = Index;
            break;
        }
    }

    int32 Count = 0;
    for (int32 Index = SentenceStart; Index <= SentenceEnd; ++Index)
    {
        if (Plan.WordBoundaryPauseClassAfter[Index]
            == EOffgridAIBoundaryPauseClass::SoftListPause)
        {
            ++Count;
        }
    }
    return Count;
}

static bool IsFirstListBoundaryInSentence(
    const FOffgridAITextVisemePlan& Plan,
    int32 BoundaryWordIndex)
{
    if (BoundaryWordIndex < 0)
        return false;

    int32 SentenceStart = 0;
    for (int32 Index = BoundaryWordIndex - 1; Index >= 0; --Index)
    {
        if (Plan.WordBoundaryPauseClassAfter.IsValidIndex(Index)
            && Plan.WordBoundaryPauseClassAfter[Index]
                == EOffgridAIBoundaryPauseClass::HardBreakPause)
        {
            SentenceStart = Index + 1;
            break;
        }
    }
    for (int32 Index = SentenceStart; Index < BoundaryWordIndex; ++Index)
    {
        if (Plan.WordBoundaryPauseClassAfter.IsValidIndex(Index)
            && Plan.WordBoundaryPauseClassAfter[Index]
                == EOffgridAIBoundaryPauseClass::SoftListPause)
        {
            return false;
        }
    }
    return Plan.WordBoundaryPauseClassAfter.IsValidIndex(BoundaryWordIndex)
        && Plan.WordBoundaryPauseClassAfter[BoundaryWordIndex]
            == EOffgridAIBoundaryPauseClass::SoftListPause;
}

static bool IsLastListBoundaryInSentence(
    const FOffgridAITextVisemePlan& Plan,
    int32 BoundaryWordIndex)
{
    if (!Plan.WordBoundaryPauseClassAfter.IsValidIndex(BoundaryWordIndex)
        || Plan.WordBoundaryPauseClassAfter[BoundaryWordIndex]
            != EOffgridAIBoundaryPauseClass::SoftListPause)
    {
        return false;
    }

    for (int32 Index = BoundaryWordIndex + 1;
        Index < Plan.WordBoundaryPauseClassAfter.Num();
        ++Index)
    {
        const EOffgridAIBoundaryPauseClass PauseClass =
            Plan.WordBoundaryPauseClassAfter[Index];
        if (PauseClass == EOffgridAIBoundaryPauseClass::SoftListPause)
            return false;
        if (PauseClass == EOffgridAIBoundaryPauseClass::HardBreakPause)
            break;
    }
    return true;
}

static bool FindTrailingListQuietRun(
    const FOffgridAILipsyncRuntimeUpdateInput& Input,
    float SearchStartSec,
    float& OutQuietStartSec)
{
    if (!Input.AudioFeatureFrames)
        return false;

    float QuietStartSec = -1.0f;
    for (const auto& Frame : *Input.AudioFeatureFrames)
    {
        if (Frame.AudioBufferCenterSec < SearchStartSec - 0.001f)
            continue;
        if (Frame.AudioBufferCenterSec > Input.ObservedAudioBufferEndSec + 0.001f)
            break;
        if (!Frame.bLearnedSpeech)
        {
            if (QuietStartSec < 0.0f)
                QuietStartSec = Frame.AudioBufferCenterSec;
        }
        else
        {
            QuietStartSec = -1.0f;
        }
    }
    OutQuietStartSec = QuietStartSec;
    return QuietStartSec >= 0.0f;
}

static bool FindProsodicRestart(
    const FOffgridAILipsyncRuntimeUpdateInput& Input,
    float SearchStartSec,
    float LastConsumedSec,
    float MaximumValleyRMSNorm,
    float MinimumReboundRatio,
    float& OutRestartSec)
{
    if (!Input.AudioFeatureFrames || Input.AudioFeatureFrames->Num() < 5)
        return false;

    const TArray<FOffgridAIStreamingAudioFeatureFrame>& Frames =
        *Input.AudioFeatureFrames;
    for (int32 ValleyIndex = 0; ValleyIndex < Frames.Num(); ++ValleyIndex)
    {
        const auto& Valley = Frames[ValleyIndex];
        if (!Valley.bLocalRMSValley
            || Valley.AudioBufferCenterSec < SearchStartSec - 0.001f
            || Valley.AudioBufferCenterSec <= LastConsumedSec + 0.001f
            || Valley.RMSNorm > MaximumValleyRMSNorm)
            continue;

        const float ValleyRMS = FMath::Max(Valley.RMS, 0.0000001f);
        float PeakFlux = 0.0f;
        int32 FirstRestartIndex = INDEX_NONE;
        for (int32 FrameIndex = ValleyIndex + 1; FrameIndex < Frames.Num(); ++FrameIndex)
        {
            const auto& Frame = Frames[FrameIndex];
            const float DelaySec = Frame.AudioBufferCenterSec - Valley.AudioBufferCenterSec;
            if (DelaySec > ListRestartMaximumValleySec + 0.001f)
                break;
            PeakFlux = FMath::Max(PeakFlux, Frame.Flux);
            if (FirstRestartIndex == INDEX_NONE
                && DelaySec >= ListRestartMinimumValleySec - 0.001f
                && Frame.bLearnedSpeech
                && Frame.RMS >= ValleyRMS * MinimumReboundRatio
                && (Frame.bStrongOnsetAnchor || Frame.SpeechEvidence >= 0.230f
                    || Frame.Flux >= 0.040f))
            {
                FirstRestartIndex = FrameIndex;
            }
        }
        if (FirstRestartIndex != INDEX_NONE
            && PeakFlux >= ListRestartMinimumFlux)
        {
            const auto& Restart = Frames[FirstRestartIndex];
            if (Restart.AudioBufferCenterSec <= Input.ObservedAudioBufferEndSec + 0.001f)
            {
                OutRestartSec = Restart.AudioBufferCenterSec;
                return true;
            }
        }
    }
    return false;
}

static void UpdateSyllableAnchor(
    const FOffgridAILipsyncRuntimeUpdateInput& Input,
    const FRuntimePrior& Prior,
    int32 NextPhoneIndex,
    int32 ActiveAudioRegionIndex,
    int32 ActiveTextRegionIndex,
    const FOffgridAIStreamingSpeechRegion& ActiveAudioRegion,
    FOffgridAIBoundaryPlaybackState& State)
{
    if (!Input.TextPlan || !Input.AudioFeatureFrames || !Input.SpeechRegions
        || Input.AudioFeatureFrames->Num() < 5)
        return;

    const int32 FeatureFrameCount = Input.AudioFeatureFrames->Num();
    if (FeatureFrameCount <= State.LastAnalyzedFeatureFrameCount
        || FeatureFrameCount - State.LastAnalyzedFeatureFrameCount < 3)
        return;
    State.LastAnalyzedFeatureFrameCount = FeatureFrameCount;

    FOffgridAIStreamingEvidenceSurfaceConfig Config;
    Config.PrerollSec = FMath::Max(Input.PrerollSec, 0.100f);
    Config.PostrollSec = 1.500f;
    const TArray<FOffgridAIAudioLandmarkObservation> Evidence =
        FOffgridAIStreamingEvidenceSurface::Analyze(*Input.AudioFeatureFrames, Config);

    if (!Input.bEnableFocusedWordStartAlignment)
    {
        const TArray<FOffgridAIStreamingSyllablePositionEstimate> HistoricalAnchors =
            FOffgridAIStreamingSyllablePositionEstimator::EstimateHistoricalAnchors(
                *Input.TextPlan,
                Evidence,
                *Input.SpeechRegions,
                2,
                MinAnchorMatchScore);
        const float HistoricalRegionEnd = ActiveAudioRegion.bEnded
            ? CausalRegionEnd(ActiveAudioRegion)
            : Input.ObservedAudioBufferEndSec;
        const FOffgridAIStreamingSyllablePositionEstimate* Historical = nullptr;
        for (const auto& Anchor : HistoricalAnchors)
        {
            if (Anchor.DecisionSec > Input.ObservedAudioBufferEndSec + 0.001f
                || Anchor.SyllableIndex <= State.LastMatchedSyllableIndex
                || Anchor.NucleusPhoneIndex > NextPhoneIndex + 6
                || Anchor.AudioCenterSec < ActiveAudioRegion.AudioBufferStartSec - 0.001f
                || Anchor.AudioCenterSec > HistoricalRegionEnd + 0.001f)
                continue;
            if (!Historical || Anchor.SyllableIndex > Historical->SyllableIndex)
                Historical = &Anchor;
        }
        if (!Historical || !Prior.PhoneCenters.IsValidIndex(Historical->NucleusPhoneIndex))
            return;

        const float PriorCenter = Prior.PhoneCenters[Historical->NucleusPhoneIndex];
        const float PredictedAudioCenter = PriorToAudio(State, PriorCenter);
        const float CorrectionSec = FMath::Clamp(
            Historical->AudioCenterSec - PredictedAudioCenter,
            -MaxSyllableRebaseSec,
            MaxSyllableRebaseSec);
        State.TimelinePriorAnchorSec = PriorCenter;
        State.TimelineAudioAnchorSec = PredictedAudioCenter + CorrectionSec;
        State.LastMatchedSyllableIndex = Historical->SyllableIndex;
        State.LastMatchedSyllablePhoneIndex = Historical->NucleusPhoneIndex;
        State.LastMatchedSyllableAudioSec = Historical->AudioCenterSec;
        State.LastMatchedSyllableConfidence = Historical->Confidence;
        ++State.BoundedSyllableRebaseCount;

        FOffgridAIRuntimeSyllableAssignmentDiagnosticRow Row;
        Row.LineID = Input.LineID;
        Row.AudioSpeechRegionIndex = ActiveAudioRegionIndex;
        Row.TextSpeechRegionIndex = ActiveTextRegionIndex;
        Row.PhoneIndex = Historical->NucleusPhoneIndex;
        Row.ObservedAudioSec = Historical->AudioCenterSec;
        Row.Prominence = Historical->MatchScore;
        Row.Confidence = Historical->Confidence;
        Row.SkipCount = FMath::Max(Historical->NucleusPhoneIndex - NextPhoneIndex, 0);
        Row.AnchorKind = FName(TEXT("bounded_syllable_rebase"));
        Row.TimelineCorrectionSec = CorrectionSec;
        State.PendingSyllableAssignments.Add(Row);
        return;
    }

    const TArray<FOffgridAIStreamingSyllablePositionEstimate> Anchors =
        FOffgridAIStreamingSyllablePositionEstimator::EstimateHistoricalAnchors(
            *Input.TextPlan,
            Evidence,
            *Input.SpeechRegions,
            1,
            MinAnchorMatchScore);

    const float RegionEnd = ActiveAudioRegion.bEnded
        ? CausalRegionEnd(ActiveAudioRegion)
        : Input.ObservedAudioBufferEndSec;
    const FOffgridAIStreamingSyllablePositionEstimate* Selected = nullptr;
    for (const FOffgridAIStreamingSyllablePositionEstimate& Anchor : Anchors)
    {
        if (Anchor.DecisionSec > Input.ObservedAudioBufferEndSec + 0.001f
            || Anchor.SyllableIndex <= State.LastMatchedSyllableIndex
            || Anchor.NucleusPhoneIndex > NextPhoneIndex + 6
            || Anchor.MatchScore < MinAnchorMatchScore
            || Anchor.SpeechRegionIndex != ActiveTextRegionIndex
            || Anchor.AudioCenterSec < ActiveAudioRegion.AudioBufferStartSec - 0.001f
            || Anchor.AudioCenterSec > RegionEnd + 0.001f)
            continue;
        if (!Selected || Anchor.SyllableIndex > Selected->SyllableIndex)
            Selected = &Anchor;
    }
    if (!Selected)
    {
        State.PendingMatchedSyllableIndex = INDEX_NONE;
        State.PendingMatchedSyllableAudioSec = -1.0f;
        State.PendingMatchedStableUpdates = 0;
        return;
    }
    const FOffgridAIStreamingSyllablePositionEstimate& SelectedAnchor = *Selected;
    if (!Prior.PhoneCenters.IsValidIndex(SelectedAnchor.NucleusPhoneIndex)
        || SelectedAnchor.NucleusPhoneIndex > NextPhoneIndex + 6)
        return;

    const float PriorCenter = Prior.PhoneCenters[SelectedAnchor.NucleusPhoneIndex];
    const float PredictedAudioCenter = PriorToAudio(State, PriorCenter);
    const float CorrectionSec = FMath::Clamp(
        SelectedAnchor.AudioCenterSec - PredictedAudioCenter,
        -MaxSyllableRebaseSec,
        MaxSyllableRebaseSec);
    float UpdatedRate = State.TimelineRate;
    const int32 NextWordIndex = Input.TextPlan->ExpectedPhones.IsValidIndex(NextPhoneIndex)
        ? Input.TextPlan->ExpectedPhones[NextPhoneIndex].WordIndex
        : INDEX_NONE;
    const bool bDenseListSentence =
        ListBoundaryCountInSentence(*Input.TextPlan, NextWordIndex)
            >= DenseListMinimumBoundaryCount;
    if (!bDenseListSentence
        && Prior.PhoneCenters.IsValidIndex(State.LastMatchedSyllablePhoneIndex)
        && State.LastMatchedSyllableAudioSec
            >= ActiveAudioRegion.AudioBufferStartSec - 0.001f)
    {
        const float PriorSpanSec = PriorCenter
            - Prior.PhoneCenters[State.LastMatchedSyllablePhoneIndex];
        const float AudioSpanSec = SelectedAnchor.AudioCenterSec
            - State.LastMatchedSyllableAudioSec;
        if (PriorSpanSec >= FocusedMinimumRateSpanSec
            && AudioSpanSec >= FocusedMinimumRateSpanSec)
        {
            const float MeasuredRate = FMath::Clamp(
                PriorSpanSec / AudioSpanSec,
                FocusedMinimumAdaptiveRate,
                FocusedMaximumAdaptiveRate);
            UpdatedRate = FMath::Clamp(
                State.TimelineRate
                    + (MeasuredRate - State.TimelineRate)
                        * FocusedAdaptiveRateBlend,
                FocusedMinimumAdaptiveRate,
                FocusedMaximumAdaptiveRate);
        }
    }

    State.TimelinePriorAnchorSec = PriorCenter;
    State.TimelineAudioAnchorSec = PredictedAudioCenter + CorrectionSec;
    State.TimelineRate = UpdatedRate;
    State.LastMatchedSyllableIndex = SelectedAnchor.SyllableIndex;
    State.LastMatchedSyllablePhoneIndex = SelectedAnchor.NucleusPhoneIndex;
    State.LastMatchedSyllableAudioSec = SelectedAnchor.AudioCenterSec;
    State.LastMatchedSyllableConfidence = SelectedAnchor.Confidence;
    State.PendingMatchedSyllableIndex = INDEX_NONE;
    State.PendingMatchedSyllableAudioSec = -1.0f;
    State.PendingMatchedStableUpdates = 0;
    ++State.BoundedSyllableRebaseCount;

    FOffgridAIRuntimeSyllableAssignmentDiagnosticRow Row;
    Row.LineID = Input.LineID;
    Row.AudioSpeechRegionIndex = ActiveAudioRegionIndex;
    Row.TextSpeechRegionIndex = ActiveTextRegionIndex;
    Row.PhoneIndex = SelectedAnchor.NucleusPhoneIndex;
    Row.ObservedAudioSec = SelectedAnchor.AudioCenterSec;
    Row.Prominence = SelectedAnchor.MatchScore;
    Row.Confidence = SelectedAnchor.Confidence;
    Row.SkipCount = FMath::Max(SelectedAnchor.NucleusPhoneIndex - NextPhoneIndex, 0);
    Row.AnchorKind = FName(TEXT("bounded_syllable_rebase"));
    Row.TimelineCorrectionSec = CorrectionSec;
    State.PendingSyllableAssignments.Add(Row);
}

static void FillCommittedEvent(
    const FOffgridAILipsyncRuntimeUpdateInput& Input,
    const FOffgridAITextVisemePlan& Plan,
    const FRuntimePrior& Prior,
    const FOffgridAIBoundaryPlaybackState& State,
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
    Out.bUsedResumeAnchor = RegionIndex > 0
        && State.LastAnchoredSpeechRegionIndex == RegionIndex;
    const bool bSyllableAnchor = CommitReason == FName(TEXT("bounded_syllable_rebase_commit"));
    Out.AcousticAnchorKind = bSyllableAnchor
        ? FName(TEXT("syllable_pulse"))
        : FName(TEXT("speech_region"));
    Out.AcousticAnchorSeconds = bSyllableAnchor
        ? State.LastMatchedSyllableAudioSec
        : State.TimelineAudioAnchorSec;
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
    bEnableFocusedWordStartAlignment = false;
    bEnableAudioPulseMouthExperiment = false;
    bEnableSyllablePacedVisemesExperiment = false;
    PlaybackSec = 0.0f;
    bBegun = false;
    bPlaybackStarted = false;
    bCommittedTrackBuilt = false;
    bInputStreamClosed = false;
    TextPlan = FOffgridAITextVisemePlan();
    Detector.Reset();
    ResolvedSpeechRegions.Reset();
    CommittedTrack = FOffgridAICommittedVisemeTrack();
    RuntimeCommitDiagnosticRows.Reset();
    RuntimeSpeechRegionDiagnosticRows.Reset();
    RuntimeBoundaryDiagnosticRows.Reset();
    RuntimeSyllableAssignmentDiagnosticRows.Reset();
    RuntimeCommitDiagnosticUpdateOrdinal = 0;
    StreamTailDiagnosticRow = FOffgridAIStreamTailDiagnosticRow();
    PCMChunkCount = 0;
    PCMBytesReceived = 0;
    PCMSamplesReceived = 0;
    LastPCMChunkSampleRate = 0;
    LastPCMChunkChannels = 0;
    LastPCMChunkStartSample = -1;
    LastPCMChunkEndSample = -1;
    PlaybackState = FOffgridAIBoundaryPlaybackState();
}

void FOffgridAILipsyncRuntimeSession::BeginLine(const FOffgridAILipsyncRuntimeBeginInput& Input)
{
    Reset();
    NPCID = Input.NPCID;
    LineID = Input.LineID;
    DialogueText = Input.DialogueText;
    PrerollSec = FMath::Max(Input.PrerollSec, 0.0f);
    bEnableFocusedWordStartAlignment = Input.bEnableFocusedWordStartAlignment;
    bEnableAudioPulseMouthExperiment = Input.bEnableAudioPulseMouthExperiment;
    bEnableSyllablePacedVisemesExperiment = Input.bEnableSyllablePacedVisemesExperiment;
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
    // Experimental audio-driven controllers deliberately keep pause/resume
    // ownership entirely in the PCM detector. The production focused
    // scheduler may still lower the quiet-run threshold inside recognized
    // transcript lists, but punctuation must not influence experimental
    // speech-region topology.
    const bool bUseTranscriptConditionedListSensitivity =
        bEnableFocusedWordStartAlignment
        && !bEnableAudioPulseMouthExperiment
        && !bEnableSyllablePacedVisemesExperiment;
    Detector.SetListGapSensitivity(
        bUseTranscriptConditionedListSensitivity
        && IsRecognizedListActive(TextPlan, PlaybackState));
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
    UpdatePlaybackGate(Detector.GetObservedAudioBufferEndSec());
    RefreshResolvedSpeechRegions();
    FOffgridAILipsyncRuntimeUpdateInput Input;
    Input.TextPlan = &TextPlan;
    Input.SpeechRegions = &ResolvedSpeechRegions;
    Input.GapCandidates = &Detector.GetGapCandidates();
    Input.SoftLullCandidates = &Detector.GetSoftLullCandidates();
    Input.AudioFeatureFrames = &Detector.GetFeatureFrames();
    Input.CurrentPlaybackSec = PlaybackSec;
    Input.PrerollSec = PrerollSec;
    Input.ObservedAudioBufferEndSec = Detector.GetObservedAudioBufferEndSec();
    Input.bEnableFocusedWordStartAlignment = bEnableFocusedWordStartAlignment;
    Input.bEnableAudioPulseMouthExperiment = bEnableAudioPulseMouthExperiment;
    Input.bEnableSyllablePacedVisemesExperiment = bEnableSyllablePacedVisemesExperiment;
    Input.bInputStreamClosed = bInputStreamClosed;
    Input.NPCID = NPCID;
    Input.LineID = LineID;
    FOffgridAILipsyncRuntimeAdapter::UpdateCommittedTrack(
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
    Input.GapCandidates = &Detector.GetGapCandidates();
    Input.SoftLullCandidates = &Detector.GetSoftLullCandidates();
    Input.AudioFeatureFrames = &Detector.GetFeatureFrames();
    Input.CurrentPlaybackSec = PlaybackSec;
    Input.PrerollSec = PrerollSec;
    Input.ObservedAudioBufferEndSec = Detector.GetObservedAudioBufferEndSec();
    Input.bEnableFocusedWordStartAlignment = bEnableFocusedWordStartAlignment;
    Input.bEnableAudioPulseMouthExperiment = bEnableAudioPulseMouthExperiment;
    Input.bEnableSyllablePacedVisemesExperiment = bEnableSyllablePacedVisemesExperiment;
    Input.bInputStreamClosed = true;
    Input.bPlaybackFinalized = true;
    Input.NPCID = NPCID;
    Input.LineID = LineID;
    FOffgridAILipsyncRuntimeAdapter::UpdateCommittedTrack(
        Input, CommittedTrack, PlaybackState, bCommittedTrackBuilt);
    RecordRuntimeDiagnostics(PlaybackSec, true);
}

void FOffgridAILipsyncRuntimeSession::UpdatePlaybackGate(float ObservedEndSec)
{
    if (bPlaybackStarted) return;
    if (Detector.HasObservedFirstSpeechStart() || ObservedEndSec >= PrerollSec || bInputStreamClosed)
        bPlaybackStarted = true;
}

void FOffgridAILipsyncRuntimeSession::RefreshResolvedSpeechRegions()
{
    ResolvedSpeechRegions = Detector.GetRefinedSpeechRegions();
}

void FOffgridAILipsyncRuntimeSession::RecordRuntimeDiagnostics(
    float CurrentPlaybackSec,
    bool bFinalReplay)
{
    ++RuntimeCommitDiagnosticUpdateOrdinal;
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
        Row.UpdateOrdinal = RuntimeCommitDiagnosticUpdateOrdinal;
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
    StateRow.UpdateOrdinal = RuntimeCommitDiagnosticUpdateOrdinal;
    StateRow.bFinalReplay = bFinalReplay;
    StateRow.CurrentPlaybackSec = CurrentPlaybackSec;
    StateRow.bPlayheadStarted = PlaybackState.bPlayheadStarted;
    StateRow.ActiveSpeechRegionIndex = PlaybackState.ActiveSpeechRegionIndex;
    StateRow.ActiveTextSpeechRegionIndex = PlaybackState.ActiveTextSpeechRegionIndex;
    StateRow.ActiveTextRegionAudioStartSec = PlaybackState.ActiveTextRegionAudioStartSec;
    if (ResolvedSpeechRegions.IsValidIndex(PlaybackState.ActiveSpeechRegionIndex))
    {
        const auto& Region = ResolvedSpeechRegions[PlaybackState.ActiveSpeechRegionIndex];
        StateRow.ActiveRegionStartSec = Region.AudioBufferStartSec;
        StateRow.ActiveRegionEndSec = CausalRegionEnd(Region);
        StateRow.bAudioSpeechActive = CurrentPlaybackSec >= Region.AudioBufferStartSec
            && (!Region.bEnded || CurrentPlaybackSec <= CausalRegionEnd(Region));
    }
    StateRow.TimelinePriorAnchorSec = PlaybackState.TimelinePriorAnchorSec;
    StateRow.TimelineAudioAnchorSec = PlaybackState.TimelineAudioAnchorSec;
    StateRow.TimelineRate = PlaybackState.TimelineRate;
    StateRow.LastMatchedSyllableIndex = PlaybackState.LastMatchedSyllableIndex;
    StateRow.LastMatchedSyllablePhoneIndex = PlaybackState.LastMatchedSyllablePhoneIndex;
    StateRow.LastMatchedSyllableAudioSec = PlaybackState.LastMatchedSyllableAudioSec;
    StateRow.LastMatchedSyllableConfidence = PlaybackState.LastMatchedSyllableConfidence;
    StateRow.BoundedSyllableRebaseCount = PlaybackState.BoundedSyllableRebaseCount;
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
        Row.UpdateOrdinal = RuntimeCommitDiagnosticUpdateOrdinal;
        RuntimeSyllableAssignmentDiagnosticRows.Add(Row);
    }
    PlaybackState.PendingSyllableAssignments.Reset();

    while (RuntimeCommitDiagnosticRows.Num() < CommittedTrack.Events.Num())
    {
        const auto& Event = CommittedTrack.Events[RuntimeCommitDiagnosticRows.Num()];
        FOffgridAIRuntimeCommitDiagnosticRow Row;
        Row.LineID = LineID;
        Row.UpdateOrdinal = RuntimeCommitDiagnosticUpdateOrdinal;
        Row.bFinalReplay = bFinalReplay;
        Row.CurrentPlaybackSec = CurrentPlaybackSec;
        Row.PrerollSec = PrerollSec;
        Row.SourceEventIndex = Event.EventIndex;
        Row.Word = Event.SourceWord;
        Row.PoseID = Event.PoseID;
        Row.PlannedCenterSec = Event.PriorCenterSeconds;
        Row.CommittedCenterSec = Event.FinalRenderCenterSeconds;
        Row.RenderStartSec = Event.RenderStartSeconds;
        Row.RenderEndSec = Event.RenderEndSeconds;
        Row.CommitReason = Event.CommitReason;
        Row.PlaybackMode = FName(TEXT("audio_primary"));
        Row.bMappedToObservedSpeech = Event.bMappedToObservedSpeech;
        Row.bUsedInitialSpeechAnchor = Event.bUsedInitialSpeechAnchor;
        Row.bUsedResumeAnchor = Event.bUsedResumeAnchor;
        Row.AcousticAnchorKind = Event.AcousticAnchorKind;
        Row.AcousticAnchorSec = Event.AcousticAnchorSeconds;
        Row.AcousticAnchorErrorSec = Event.AcousticAnchorErrorSeconds;
        Row.DiagnosticKind = FName(TEXT("commit"));
        RuntimeCommitDiagnosticRows.Add(Row);
    }
}

static float RefinedRegionOnsetSec(
    const FOffgridAILipsyncRuntimeUpdateInput& Input,
    const FOffgridAIStreamingSpeechRegion& Region)
{
    float AnchorSec = Region.AudioBufferStartSec;
    if (!Input.AudioFeatureFrames || Input.AudioFeatureFrames->Num() < 5) return AnchorSec;

    FOffgridAIStreamingEvidenceSurfaceConfig Config;
    Config.PrerollSec = FMath::Max(Input.PrerollSec, 0.100f);
    Config.PostrollSec = 0.250f;
    const TArray<FOffgridAIAudioLandmarkObservation> Evidence =
        FOffgridAIStreamingEvidenceSurface::Analyze(*Input.AudioFeatureFrames, Config);
    const FOffgridAIAudioLandmarkObservation* Best = nullptr;
    const FOffgridAIAudioLandmarkObservation* ForwardFallback = nullptr;
    for (const auto& Observation : Evidence)
    {
        if (Observation.Type != EOffgridAIAudioLandmarkType::Resume
            || Observation.DecisionSec > Input.ObservedAudioBufferEndSec + 0.001f)
            continue;
        const float OffsetSec = Observation.CenterSec - Region.AudioBufferStartSec;
        if (FMath::Abs(OffsetSec) <= 0.100f)
        {
            if (!Best || FMath::Abs(OffsetSec)
                    < FMath::Abs(Best->CenterSec - Region.AudioBufferStartSec))
                Best = &Observation;
        }
        else if (OffsetSec > 0.100f
            && OffsetSec <= 0.250f
            && Observation.Score >= 0.500f
            && (!ForwardFallback
                || Observation.CenterSec < ForwardFallback->CenterSec))
        {
            ForwardFallback = &Observation;
        }
    }
    if (!Best)
        Best = ForwardFallback;
    return Best ? Best->CenterSec : AnchorSec;
}

static void AnchorSimpleCursorToRegion(
    const FOffgridAILipsyncRuntimeUpdateInput& Input,
    const FOffgridAITextVisemePlan& Plan,
    const FRuntimePrior& Prior,
    int32 EventIndex,
    int32 RegionIndex,
    const FOffgridAIStreamingSpeechRegion& Region,
    FOffgridAIBoundaryPlaybackState& State)
{
    State.ActiveSpeechRegionIndex = RegionIndex;
    State.ActiveTextSpeechRegionIndex = Plan.Events.IsValidIndex(EventIndex)
        ? Plan.Events[EventIndex].SpeechRegionIndex
        : INDEX_NONE;
    const float RegionAnchorSec = Input.bEnableFocusedWordStartAlignment
        ? RefinedRegionOnsetSec(Input, Region)
        : Region.AudioBufferStartSec;
    State.ActiveTextRegionAudioStartSec = RegionAnchorSec;
    State.TimelinePriorAnchorSec = Prior.EventCenters.IsValidIndex(EventIndex)
        ? Prior.EventCenters[EventIndex]
        : Prior.TotalSec;
    if (Input.bEnableFocusedWordStartAlignment
        && Plan.Events.IsValidIndex(EventIndex))
    {
        const int32 WordIndex = Plan.Events[EventIndex].WordIndex;
        if (Plan.WordPhoneBeginIndices.IsValidIndex(WordIndex))
        {
            const int32 WordBeginPhoneIndex = Plan.WordPhoneBeginIndices[WordIndex];
            if (Prior.PhoneStarts.IsValidIndex(WordBeginPhoneIndex))
            {
                // Every observed resume is a new local timing origin. Preserve
                // only the current word's leading-phone time; rewinding to the
                // sentence-level text-region start would skip acoustic regions
                // when a list contains several real pauses.
                State.TimelinePriorAnchorSec = Prior.PhoneStarts[WordBeginPhoneIndex];
            }
        }
    }
    State.TimelineAudioAnchorSec = RegionAnchorSec;
    State.TimelineRate = Input.bEnableFocusedWordStartAlignment
        ? FocusedPriorPlaybackRate
        : NominalPriorPlaybackRate;
    State.bPlayheadStarted = true;
    // Region onset is authoritative. Stable syllable evidence may subsequently
    // move only the uncommitted suffix.
}

static void UpdateSimpleCommittedTrack(
    const FOffgridAILipsyncRuntimeUpdateInput& Input,
    FOffgridAICommittedVisemeTrack& InOutTrack,
    FOffgridAIBoundaryPlaybackState& State,
    bool& bInOutTrackBuilt)
{
    if (!Input.TextPlan || !Input.SpeechRegions) return;
    const FOffgridAITextVisemePlan& Plan = *Input.TextPlan;
    const TArray<FOffgridAIStreamingSpeechRegion>& Regions = *Input.SpeechRegions;
    const FRuntimePrior Prior = BuildRuntimePrior(Plan);

    InOutTrack.NPCID = Input.NPCID;
    InOutTrack.LineID = Input.LineID;
    InOutTrack.SpeechRegions.Reset();
    for (const auto& Region : Regions)
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
    bInOutTrackBuilt = true;
    State.SchedulerBlockReason = NAME_None;

    int32 EventIndex = FMath::Clamp(State.NextTextEventIndex, 0, Plan.Events.Num());
    if (EventIndex >= Plan.Events.Num())
    {
        State.SchedulerBlockReason = FName(TEXT("complete"));
    }
    else if (Regions.Num() == 0)
    {
        State.SchedulerBlockReason = FName(TEXT("waiting_for_speech_open"));
    }
    else
    {
        if (!State.bPlayheadStarted)
            AnchorSimpleCursorToRegion(Input, Plan, Prior, EventIndex, 0, Regions[0], State);

        float LastCenter = LastCommittedCenter(InOutTrack);
        while (EventIndex < Plan.Events.Num())
        {
            if (!Regions.IsValidIndex(State.ActiveSpeechRegionIndex))
            {
                State.SchedulerBlockReason = FName(TEXT("waiting_for_speech_resume"));
                break;
            }

            const FOffgridAIStreamingSpeechRegion& Region =
                Regions[State.ActiveSpeechRegionIndex];
            const int32 EventTextRegionIndex = Plan.Events[EventIndex].SpeechRegionIndex;
            const int32 EventWordIndex = Plan.Events[EventIndex].WordIndex;
            const bool bWordFollowsStandardComma =
                IsStandardCommaBoundary(Plan, EventWordIndex - 1);
            const bool bFirstEventInWord = EventIndex == 0
                || Plan.Events[EventIndex - 1].WordIndex != EventWordIndex;
            const int32 PreviousWordIndex = EventIndex > 0
                ? Plan.Events[EventIndex - 1].WordIndex
                : INDEX_NONE;
            const float DeclaredPauseSec =
                Plan.WordBoundaryPauseSecondsAfter.IsValidIndex(PreviousWordIndex)
                ? FMath::Max(
                    Plan.WordBoundaryPauseSecondsAfter[PreviousWordIndex],
                    0.0f)
                : 0.0f;
            const bool bDeclaredPauseBoundary = bFirstEventInWord
                && DeclaredPauseSec > 0.001f;
            const bool bPauseBoundaryNeedsResolution = bDeclaredPauseBoundary
                && State.LastResolvedPauseBoundaryWordIndex != PreviousWordIndex;
            const bool bSoftListBoundary = bFirstEventInWord
                && Plan.WordBoundaryPauseClassAfter.IsValidIndex(PreviousWordIndex)
                && Plan.WordBoundaryPauseClassAfter[PreviousWordIndex]
                    == EOffgridAIBoundaryPauseClass::SoftListPause;
            const bool bHardBreakBoundary = bFirstEventInWord
                && Plan.WordBoundaryPauseClassAfter.IsValidIndex(PreviousWordIndex)
                && Plan.WordBoundaryPauseClassAfter[PreviousWordIndex]
                    == EOffgridAIBoundaryPauseClass::HardBreakPause;
            const bool bCommaBoundary = bFirstEventInWord
                && Plan.WordBoundaryPunctuationAfter.IsValidIndex(PreviousWordIndex)
                && Plan.WordBoundaryPunctuationAfter[PreviousWordIndex] == TEXT(',');
            const bool bStrictHardBreakBoundary = bHardBreakBoundary
                && !bCommaBoundary;
            const bool bDenseListEntryBoundary = bSoftListBoundary
                && ListBoundaryCountInSentence(Plan, EventWordIndex)
                    >= DenseListMinimumBoundaryCount
                && (IsFirstListBoundaryInSentence(Plan, PreviousWordIndex)
                    || IsLastListBoundaryInSentence(Plan, PreviousWordIndex));

            // A word that has already published its first event owns that
            // runtime region. If the region closes with an uncommitted suffix,
            // finish the suffix at the owner's tail before considering any
            // successor region. This is the constructive half of the
            // fail-closed invariant below: it prevents a valid ownership guard
            // from turning a late boundary decision into a permanent stall.
            const bool bWordOwnedByActiveRegion = !bFirstEventInWord
                && InOutTrack.Events.Num() > 0
                && InOutTrack.Events.Last().WordIndex == EventWordIndex
                && InOutTrack.Events.Last().SpeechRegionIndex
                    == State.ActiveSpeechRegionIndex;
            const bool bFinalClosedRegion = Input.bInputStreamClosed
                && State.ActiveSpeechRegionIndex == Regions.Num() - 1;
            const bool bSuccessorRegionAvailable =
                Regions.IsValidIndex(State.ActiveSpeechRegionIndex + 1);
            if (bWordOwnedByActiveRegion
                && Region.bEnded
                && bSuccessorRegionAvailable)
            {
                const int32 SuffixBeginEventIndex = EventIndex;
                int32 WordEndEventIndex = EventIndex;
                while (WordEndEventIndex + 1 < Plan.Events.Num()
                    && Plan.Events[WordEndEventIndex + 1].WordIndex
                        == EventWordIndex)
                {
                    ++WordEndEventIndex;
                }
                const int32 RemainingEventCount =
                    WordEndEventIndex - EventIndex + 1;
                const float OwnerRegionEnd = CausalRegionEnd(Region);
                const float FirstCenter = FMath::Max(
                    LastCenter + MinRegionTailCompactionSpacingSec,
                    OwnerRegionEnd
                        - MinRegionTailCompactionSpacingSec
                            * static_cast<float>(RemainingEventCount - 1));
                const float LastSuffixCenter = FirstCenter
                    + MinRegionTailCompactionSpacingSec
                        * static_cast<float>(RemainingEventCount - 1);
                if (LastSuffixCenter
                    <= OwnerRegionEnd + MaxAtomicWordTailOverrunSec + 0.001f)
                {
                    while (EventIndex <= WordEndEventIndex)
                    {
                        const float Center = FirstCenter
                            + MinRegionTailCompactionSpacingSec
                                * static_cast<float>(
                                    EventIndex - SuffixBeginEventIndex);
                        FOffgridAICommittedVisemeEvent Event;
                        FillCommittedEvent(
                            Input,
                            Plan,
                            Prior,
                            State,
                            EventIndex,
                            State.ActiveSpeechRegionIndex,
                            Center,
                            FName(TEXT("owned_word_tail_commit")),
                            Event);
                        InOutTrack.Events.Add(Event);
                        LastCenter = Center;
                        ++EventIndex;
                        State.NextTextEventIndex = EventIndex;
                    }
                    continue;
                }
            }

            if (Input.bEnableFocusedWordStartAlignment
                && bSoftListBoundary
                && State.LastResolvedListBoundaryWordIndex != EventWordIndex)
            {
                if (State.PendingListBoundaryWordIndex != EventWordIndex)
                {
                    State.PendingListBoundaryWordIndex = EventWordIndex;
                    State.PendingListSearchStartSec = InOutTrack.Events.Num() > 0
                        ? InOutTrack.Events.Last().RenderEndSeconds - 0.010f
                        : Region.AudioBufferStartSec;
                    State.PendingListQuietStartSec = -1.0f;
                }
                float RestartSec = -1.0f;
                const bool bFoundRestart = FindProsodicRestart(
                        Input,
                        State.PendingListSearchStartSec,
                        State.LastConsumedProsodicRestartSec,
                        ListRestartMaximumValleyRMSNorm,
                        ListRestartMinimumReboundRatio,
                        RestartSec);
                const bool bRestartBelongsToActiveRegion = bFoundRestart
                    // Once the active decoded region has ended, a restart
                    // beyond its endpoint belongs to a later region.  Let the
                    // ordinary region handoff own that onset instead of
                    // allowing a comma boundary to steal it.
                    && (!Region.bEnded
                        || RestartSec <= CausalRegionEnd(Region) + 0.001f);
                if (bRestartBelongsToActiveRegion)
                {
                    State.TimelinePriorAnchorSec = Prior.EventCenters.IsValidIndex(EventIndex)
                        ? Prior.EventCenters[EventIndex]
                        : Prior.TotalSec;
                    State.TimelineAudioAnchorSec = RestartSec;
                    State.LastConsumedProsodicRestartSec = RestartSec;
                    State.LastListRestartWordIndex = EventWordIndex;
                    State.LastResolvedListBoundaryWordIndex = EventWordIndex;
                    State.PendingListBoundaryWordIndex = INDEX_NONE;
                    State.PendingListSearchStartSec = -1.0f;
                    State.PendingListQuietStartSec = -1.0f;
                }
                else if (bFoundRestart)
                {
                    // A later decoded region owns this landmark. Resolve the
                    // comma without rebasing so the ordinary region handoff
                    // below can anchor the word.
                    State.LastResolvedListBoundaryWordIndex = EventWordIndex;
                    State.PendingListBoundaryWordIndex = INDEX_NONE;
                    State.PendingListSearchStartSec = -1.0f;
                    State.PendingListQuietStartSec = -1.0f;
                }
                else if (bDenseListEntryBoundary)
                {
                    float QuietStartSec = -1.0f;
                    const bool bEndsInQuiet = FindTrailingListQuietRun(
                        Input,
                        State.PendingListSearchStartSec,
                        QuietStartSec);
                    if (bEndsInQuiet)
                    {
                        if (State.PendingListQuietStartSec < 0.0f)
                            State.PendingListQuietStartSec = QuietStartSec;
                        if (!Input.bInputStreamClosed)
                        {
                            State.SchedulerBlockReason = FName(
                                TEXT("waiting_for_list_prosodic_restart"));
                            break;
                        }

                        // A closed stream resolves a terminal quiet run
                        // without fabricating a pause. During live playback,
                        // persistent quiet remains pending until either a
                        // restart arrives or the detector closes the region;
                        // residual tail speech cannot release the next word.
                        State.LastResolvedListBoundaryWordIndex = EventWordIndex;
                        State.PendingListBoundaryWordIndex = INDEX_NONE;
                        State.PendingListSearchStartSec = -1.0f;
                        State.PendingListQuietStartSec = -1.0f;
                    }
                }
            }

            // Transcript identity and punctuation identify places where an
            // acoustic pause may occur; only observed audio decides whether
            // the boundary becomes a new speech region or remains continuous.
            if (Input.bEnableFocusedWordStartAlignment
                && (EventTextRegionIndex > State.ActiveTextSpeechRegionIndex
                    || bPauseBoundaryNeedsResolution))
            {
                // A sentence or short final tag can share one decoded speech
                // region with the preceding phrase. A valley and acoustic
                // reattack after the last rendered event can resolve that
                // punctuation boundary without inventing another scheduler
                // or a text-owned time. A decoded successor still takes
                // precedence once the current region has ended.
                bool bResolvedWithinFinalRegion = false;
                const int32 NextRegionIndex = State.ActiveSpeechRegionIndex + 1;
                if (Region.bEnded
                    && Input.bInputStreamClosed
                    && !Regions.IsValidIndex(NextRegionIndex))
                {
                    const float SearchStartSec = InOutTrack.Events.Num() > 0
                        ? InOutTrack.Events.Last().RenderEndSeconds - 0.010f
                        : Region.AudioBufferStartSec;
                    float RestartSec = -1.0f;
                    if (FindProsodicRestart(
                            Input,
                            SearchStartSec,
                            State.LastConsumedProsodicRestartSec,
                            0.200f,
                            2.0f,
                            RestartSec))
                    {
                        State.ActiveTextSpeechRegionIndex = EventTextRegionIndex;
                        State.LastResolvedPauseBoundaryWordIndex = PreviousWordIndex;
                        State.ActiveTextRegionAudioStartSec = RestartSec;
                        State.TimelinePriorAnchorSec = Prior.EventCenters.IsValidIndex(EventIndex)
                            ? Prior.EventCenters[EventIndex]
                            : Prior.TotalSec;
                        State.TimelineAudioAnchorSec = RestartSec;
                        State.LastConsumedProsodicRestartSec = RestartSec;
                        bResolvedWithinFinalRegion = true;
                    }
                    else
                    {
                        // With the complete stream available, absence of both
                        // a decoded successor and a qualifying reattack means
                        // the punctuation remained inside this final acoustic
                        // region. Preserve identity and finish monotonically;
                        // the final-tail path below supplies only bounded
                        // placement, never a second timing scheduler.
                        State.ActiveTextSpeechRegionIndex = EventTextRegionIndex;
                        State.LastResolvedPauseBoundaryWordIndex = PreviousWordIndex;
                        bResolvedWithinFinalRegion = true;
                    }
                }

                if (bResolvedWithinFinalRegion)
                {
                    // Continue below and let the ordinary audio frontier gate
                    // commit the reanchored suffix.
                }
                else
                {
                    if (bHardBreakBoundary)
                    {
                        // A locally learned speaking rate belongs only to the
                        // phrase that produced its anchors. Do not extrapolate
                        // it across unresolved sentence punctuation.
                        State.TimelineRate = FocusedPriorPlaybackRate;
                    }
                    const float BoundaryCandidate =
                        PriorToAudio(State, Prior.EventCenters[EventIndex]);
                    const float ContinuityProbeSec = BoundaryCandidate
                        + FMath::Max(DeclaredPauseSec, TextRegionMergeEvidenceSec);
                    bool bObservedQuietAfterProbe = false;
                    bool bObservedSpeechAfterProbe = false;
                    float ObservedResumeSec = -1.0f;
                    float ObservedQuietStartSec = -1.0f;
                    if (Input.AudioFeatureFrames)
                    {
                        for (const auto& Frame : *Input.AudioFeatureFrames)
                        {
                            if (Frame.AudioBufferCenterSec + 0.001f < ContinuityProbeSec)
                                continue;
                            if (Frame.AudioBufferCenterSec > Input.ObservedAudioBufferEndSec + 0.001f)
                                break;
                            if (!Frame.bLearnedSpeech || Frame.RMSNorm <= 0.030f)
                            {
                                bObservedQuietAfterProbe = true;
                                if (ObservedQuietStartSec < 0.0f)
                                    ObservedQuietStartSec = Frame.AudioBufferCenterSec;
                                continue;
                            }
                            const bool bStrongVirtualResume =
                                (Frame.Flux >= 0.060f && Frame.RMSNorm >= 0.020f)
                                || (Frame.RMSNorm >= 0.150f
                                    && Frame.SpeechEvidence >= 0.200f);
                            const float ContiguousQuietSec = ObservedQuietStartSec >= 0.0f
                                ? Frame.AudioBufferCenterSec - ObservedQuietStartSec
                                : 0.0f;
                            const bool bQualifiedVirtualResume =
                                bObservedQuietAfterProbe
                                && bStrongVirtualResume
                                && (!bStrictHardBreakBoundary
                                    || ContiguousQuietSec
                                        >= HardVirtualResumeMinimumQuietSec);
                            if (bObservedQuietAfterProbe && !bQualifiedVirtualResume)
                            {
                                // A weak or too-short reattack ends this quiet
                                // run. It must not remain armed and steal a
                                // later onset from the preceding speech.
                                bObservedQuietAfterProbe = false;
                                ObservedQuietStartSec = -1.0f;
                                if (bHardBreakBoundary)
                                    continue;
                            }
                            if (bStrictHardBreakBoundary && !bQualifiedVirtualResume)
                                continue;
                            bObservedSpeechAfterProbe = true;
                            if (bQualifiedVirtualResume)
                                ObservedResumeSec = Frame.AudioBufferCenterSec;
                            break;
                        }
                    }
                    if (!Region.bEnded
                        && bObservedSpeechAfterProbe
                        // Continuous speech shortly after a duration-prior
                        // probe is not sufficient to erase a sentence
                        // boundary: a short prior can otherwise commit the
                        // next sentence into the current phrase. Hard breaks
                        // require an observed quiet-to-speech restart (or a
                        // decoded region handoff).
                        && (!bStrictHardBreakBoundary || ObservedResumeSec >= 0.0f))
                    {
                    // Punctuation may propose a text region without an
                    // independently decoded region. Continuous evidence keeps
                    // the existing timing map; quiet followed by learned speech
                    // is a virtual resume and becomes the local origin for only
                    // the uncommitted suffix.
                    State.ActiveTextSpeechRegionIndex = EventTextRegionIndex;
                    State.LastResolvedPauseBoundaryWordIndex = PreviousWordIndex;
                    if (ObservedResumeSec >= 0.0f)
                    {
                        State.ActiveTextRegionAudioStartSec = ObservedResumeSec;
                        State.TimelinePriorAnchorSec = Prior.EventCenters.IsValidIndex(EventIndex)
                            ? Prior.EventCenters[EventIndex]
                            : Prior.TotalSec;
                        State.TimelineAudioAnchorSec = ObservedResumeSec;
                    }
                    }
                    else
                    {
                        if (Regions.IsValidIndex(NextRegionIndex))
                        {
                            State.LastResolvedPauseBoundaryWordIndex = PreviousWordIndex;
                            AnchorSimpleCursorToRegion(
                                Input, Plan, Prior, EventIndex, NextRegionIndex, Regions[NextRegionIndex], State);
                            continue;
                        }
                        State.SchedulerBlockReason = FName(TEXT("waiting_for_speech_resume"));
                        break;
                    }
                }
            }
            const int32 NextPhoneIndex = Plan.Events[EventIndex].SourcePhoneGlobalIndex;

            // Syllable evidence may move only the uncommitted suffix. It never
            // changes event identity, event order, or the audio gate. The
            // region-head pose is positioned from the transcript nucleus
            // below; pulse-based rebasing starts after that anchor commits.
            const bool bFirstEventInTextRegion = EventIndex == 0
                || Plan.Events[EventIndex - 1].SpeechRegionIndex
                    != EventTextRegionIndex;
            if (!Input.bEnableFocusedWordStartAlignment || !bFirstEventInTextRegion)
            {
                UpdateSyllableAnchor(
                    Input,
                    Prior,
                    NextPhoneIndex,
                    State.ActiveSpeechRegionIndex,
                    Plan.Events[EventIndex].SpeechRegionIndex,
                    Region,
                    State);
            }

            float Candidate = PriorToAudio(State, Prior.EventCenters[EventIndex]);
            Candidate = FMath::Max(Candidate, Region.AudioBufferStartSec);
            if (LastCenter >= 0.0f)
                Candidate = FMath::Max(Candidate, LastCenter + MinEventSpacingSec);

            const bool bFirstEventInAudioRegion = InOutTrack.Events.Num() == 0
                || InOutTrack.Events.Last().SpeechRegionIndex
                    != State.ActiveSpeechRegionIndex;
            if (Input.bEnableFocusedWordStartAlignment
                && bFirstEventInAudioRegion)
            {
                const auto FirstSyllable = std::find_if(
                    Plan.Syllables.begin(),
                    Plan.Syllables.end(),
                    [&](const FOffgridAIPlannedSyllable& Syllable) {
                        return Syllable.WordIndex == EventWordIndex;
                    });
                if (FirstSyllable != Plan.Syllables.end()
                    && Prior.PhoneCenters.IsValidIndex(
                        FirstSyllable->NucleusPhoneIndex))
                {
                    Candidate = FMath::Max(
                        Candidate,
                        PriorToAudio(
                            State,
                            Prior.PhoneCenters[
                                FirstSyllable->NucleusPhoneIndex]));
                }
            }

            const float RegionEnd = CausalRegionEnd(Region);

            // A closed region makes ownership decidable for every untouched
            // word, regardless of punctuation. Require its complete projected
            // viseme sequence to fit before committing the first event;
            // otherwise move the whole word to the decoded successor. Pause
            // classes influence boundary detection, never word indivisibility.
            if (bFirstEventInWord && Region.bEnded)
            {
                int32 WordEndEventIndex = EventIndex;
                while (WordEndEventIndex + 1 < Plan.Events.Num()
                    && Plan.Events[WordEndEventIndex + 1].WordIndex
                        == EventWordIndex)
                {
                    ++WordEndEventIndex;
                }
                float WordEndCandidate = PriorToAudio(
                    State,
                    Prior.EventCenters[WordEndEventIndex]);
                WordEndCandidate = FMath::Max(
                    WordEndCandidate,
                    Candidate + MinEventSpacingSec
                        * static_cast<float>(WordEndEventIndex - EventIndex));
                if (WordEndCandidate > RegionEnd - 0.010f)
                {
                    const int32 NextRegionIndex = State.ActiveSpeechRegionIndex + 1;
                    if (Regions.IsValidIndex(NextRegionIndex))
                    {
                        AnchorSimpleCursorToRegion(
                            Input,
                            Plan,
                            Prior,
                            EventIndex,
                            NextRegionIndex,
                            Regions[NextRegionIndex],
                            State);
                        continue;
                    }

                    if (!Input.bInputStreamClosed)
                    {
                        // The current region has conclusively closed, but its
                        // successor has not opened yet. Do not publish a
                        // prefix of this word into the old region while
                        // waiting: keep the complete word untouched until the
                        // resume makes its ownership decidable. A definitively
                        // closed input instead falls through to the bounded
                        // final-tail completion path below.
                        State.SchedulerBlockReason = FName(
                            TEXT("whole_word_waiting_for_speech_resume"));
                        break;
                    }
                }
            }

            bool bUsedRegionTailCompaction = false;
            if (Input.bEnableFocusedWordStartAlignment
                && Region.bEnded
                && Candidate > RegionEnd - 0.010f)
            {
                int32 RemainingCompactionEventCount = 0;
                for (int32 TailIndex = EventIndex; TailIndex < Plan.Events.Num(); ++TailIndex)
                {
                    if (!bFinalClosedRegion
                        && Plan.Events[TailIndex].WordIndex != EventWordIndex)
                        break;
                    ++RemainingCompactionEventCount;
                }
                const bool bWordAlreadyCommittedToActiveRegion =
                    InOutTrack.Events.Num() > 0
                    && InOutTrack.Events.Last().WordIndex == EventWordIndex
                    && InOutTrack.Events.Last().SpeechRegionIndex
                        == State.ActiveSpeechRegionIndex;
                const float TailCompactionSpacing = MinRegionTailCompactionSpacingSec;
                float CompactedCandidate = RegionEnd
                    + (bFinalClosedRegion ? 0.0f : MaxRegionTailCompactionOverrunSec)
                    - TailCompactionSpacing
                        * static_cast<float>(FMath::Max(RemainingCompactionEventCount - 1, 0));
                const float EarliestCandidate = LastCenter >= 0.0f
                    ? LastCenter + TailCompactionSpacing
                    : Region.AudioBufferStartSec;
                if (bFinalClosedRegion)
                    CompactedCandidate = FMath::Max(CompactedCandidate, EarliestCandidate);
                else if (bWordAlreadyCommittedToActiveRegion
                    && EarliestCandidate
                        <= RegionEnd + MaxAtomicWordTailOverrunSec + 0.001f)
                {
                    CompactedCandidate = FMath::Max(CompactedCandidate, EarliestCandidate);
                }
                // Before the final closed region, tail compaction is only a
                // recovery for one stranded terminal pose in the current
                // word. Compressing a larger live suffix can consume a later
                // word that belongs to the next acoustic region. Once the
                // complete stream proves there is no successor, the same
                // bounded monotonic path may finish the final suffix.
                if ((RemainingCompactionEventCount == 1
                        || bWordAlreadyCommittedToActiveRegion
                        || bFinalClosedRegion)
                    && CompactedCandidate >= EarliestCandidate - 0.001f
                    && CompactedCandidate >= Region.AudioBufferStartSec - 0.001f
                    && (!bFinalClosedRegion
                        || CompactedCandidate <= Input.ObservedAudioBufferEndSec + 0.001f))
                {
                    Candidate = FMath::Max(CompactedCandidate, EarliestCandidate);
                    bUsedRegionTailCompaction = true;
                }
            }
            if (Region.bEnded
                && Candidate > RegionEnd - 0.010f
                && !bUsedRegionTailCompaction)
            {
                const int32 NextRegionIndex = State.ActiveSpeechRegionIndex + 1;
                if (!Regions.IsValidIndex(NextRegionIndex))
                {
                    State.SchedulerBlockReason = Input.bInputStreamClosed
                        ? FName(TEXT("final_speech_closed_with_unplayed_suffix"))
                        : FName(TEXT("waiting_for_speech_resume"));
                    break;
                }
                AnchorSimpleCursorToRegion(
                    Input, Plan, Prior, EventIndex, NextRegionIndex, Regions[NextRegionIndex], State);
                continue;
            }

            const float KnownSpeechFrontier = Region.bEnded
                ? (bUsedRegionTailCompaction ? Candidate : RegionEnd)
                : FMath::Min(
                    FMath::Max(
                        Region.AudioBufferLastSpeechSec + 0.030f,
                        Region.AudioBufferStartSec),
                    Input.CurrentPlaybackSec
                        + (bWordFollowsStandardComma
                            ? StandardCommaMaxCommitLeadSec
                            : MaxCommitLeadSec));
            State.SchedulerNextEventIndex = EventIndex;
            State.SchedulerNextPhoneIndex = NextPhoneIndex;
            State.SchedulerCandidateCenterSec = Candidate;
            State.SchedulerCommitFrontierSec = KnownSpeechFrontier;
            State.SchedulerCommitLeadSec = Candidate - Input.CurrentPlaybackSec;

            if (Candidate > KnownSpeechFrontier + 0.001f)
            {
                State.SchedulerBlockReason = FName(TEXT("waiting_for_audio_frontier"));
                break;
            }

            if (Candidate < Input.CurrentPlaybackSec - 0.080f)
            {
                const float RecoveryCenter = Input.CurrentPlaybackSec + MinLiveLeadSec;
                State.TimelineAudioAnchorSec += RecoveryCenter - Candidate;
                Candidate = RecoveryCenter;
                if (LastCenter >= 0.0f)
                    Candidate = FMath::Max(Candidate, LastCenter + MinEventSpacingSec);
            }

            FName Reason = FName(TEXT("duration_prior_commit"));
            if (bSoftListBoundary
                && State.LastListRestartWordIndex == EventWordIndex)
                Reason = FName(TEXT("list_prosodic_restart_commit"));
            else if (bUsedRegionTailCompaction)
                Reason = FName(TEXT("region_tail_compaction_commit"));
            else if (EventIndex == 0)
                Reason = FName(TEXT("initial_region_anchor_commit"));
            else if (FMath::Abs(Candidate - Region.AudioBufferStartSec) < 0.050f)
                Reason = FName(TEXT("resume_region_anchor_commit"));
            else if (NextPhoneIndex == State.LastMatchedSyllablePhoneIndex)
                Reason = FName(TEXT("bounded_syllable_rebase_commit"));

            // Word ownership is immutable once its first event commits. The
            // admission logic above should always resolve the complete word
            // before a region handoff. Fail closed if any future scheduling
            // path attempts to publish a suffix into another region.
            if (InOutTrack.Events.Num() > 0
                && InOutTrack.Events.Last().WordIndex == EventWordIndex
                && InOutTrack.Events.Last().SpeechRegionIndex
                    != State.ActiveSpeechRegionIndex)
            {
                State.SchedulerBlockReason = FName(
                    TEXT("word_region_invariant_guard"));
                break;
            }

            FOffgridAICommittedVisemeEvent Event;
            FillCommittedEvent(
                Input,
                Plan,
                Prior,
                State,
                EventIndex,
                State.ActiveSpeechRegionIndex,
                Candidate,
                Reason,
                Event);
            InOutTrack.Events.Add(Event);
            LastCenter = Candidate;
            ++EventIndex;
            State.NextTextEventIndex = EventIndex;
        }
    }

    State.SchedulerNextEventIndex = EventIndex;
    State.SchedulerNextPhoneIndex = Plan.Events.IsValidIndex(EventIndex)
        ? Plan.Events[EventIndex].SourcePhoneGlobalIndex
        : INDEX_NONE;
    if (Regions.Num() > 0)
    {
        InOutTrack.SpeechStartSeconds = Regions[0].AudioBufferStartSec;
        InOutTrack.SpeechEndSeconds = Regions.Last().AudioBufferEndSec;
    }
}

static float FindInterNucleusValley(
    const FOffgridAILipsyncRuntimeUpdateInput& Input,
    float PreviousNucleusSec,
    float NextNucleusSec)
{
    // We only learn that a gap is inter-nuclear once the following pulse is
    // stable. Never turn that retrospective observation into an event behind
    // the live playhead: select the quietest still-renderable point instead.
    constexpr float MinCommitLeadSec = 0.030f;
    constexpr float MinBeatSeparationSec = 0.025f;
    const float SearchStartSec = FMath::Max(
        PreviousNucleusSec + 0.020f,
        Input.CurrentPlaybackSec + MinCommitLeadSec);
    const float SearchEndSec = NextNucleusSec - MinBeatSeparationSec;
    if (SearchEndSec <= SearchStartSec)
        return -1.0f;

    float BestSec = FMath::Clamp(
        NextNucleusSec - 0.060f,
        SearchStartSec,
        SearchEndSec);
    float BestScore = TNumericLimits<float>::Max();
    if (!Input.AudioFeatureFrames)
        return BestSec;

    for (const auto& Frame : *Input.AudioFeatureFrames)
    {
        if (Frame.AudioBufferCenterSec < SearchStartSec
            || Frame.AudioBufferCenterSec > SearchEndSec)
            continue;
        const float ValleyScore = Frame.RMSNorm * 0.70f
            + Frame.SpeechEvidence * 0.30f;
        // Prefer the later frame when equally quiet. It remains visibly before
        // the next beat while maximizing the causal presentation lead.
        if (ValleyScore < BestScore - KINDA_SMALL_NUMBER
            || (FMath::Abs(ValleyScore - BestScore) <= KINDA_SMALL_NUMBER
                && Frame.AudioBufferCenterSec > BestSec))
        {
            BestScore = ValleyScore;
            BestSec = Frame.AudioBufferCenterSec;
        }
    }
    return BestSec;
}

static void UpdateAudioPulseMouthTrack(
    const FOffgridAILipsyncRuntimeUpdateInput& Input,
    FOffgridAICommittedVisemeTrack& InOutTrack,
    FOffgridAIBoundaryPlaybackState& State,
    bool& bInOutTrackBuilt)
{
    if (!Input.AudioFeatureFrames || !Input.SpeechRegions) return;

    InOutTrack.NPCID = Input.NPCID;
    InOutTrack.LineID = Input.LineID;
    InOutTrack.bAudioPulseMouthExperiment = true;
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
    FOffgridAIStreamingEvidenceSurfaceConfig Config;
    Config.PrerollSec = FMath::Max(Input.PrerollSec, 0.100f);
    Config.PostrollSec = 0.250f;
    Config.bPermissivePhoneCandidates = true;
    Config.bNucleusBeatIndicatorTuning = true;
    Config.SpeechRegions = Input.SpeechRegions;
    const TArray<FOffgridAIAudioLandmarkObservation> Evidence =
        FOffgridAIStreamingEvidenceSurface::Analyze(*Input.AudioFeatureFrames, Config);

    // A lull is a presentation gate, not a competing scheduler. Its endpoint
    // advances causally until the evidence surface observes a resume.
    InOutTrack.AcousticLulls.Reset();
    for (int32 Index = 0; Index < Evidence.Num(); ++Index)
    {
        if (Evidence[Index].Type != EOffgridAIAudioLandmarkType::Lull
            || Evidence[Index].DecisionSec > Input.ObservedAudioBufferEndSec + 0.001f)
            continue;

        FOffgridAICommittedVisemeTrack::FAcousticLull Lull;
        Lull.StartSeconds = Evidence[Index].CenterSec;
        Lull.EndSeconds = Input.ObservedAudioBufferEndSec;
        for (int32 Next = Index + 1; Next < Evidence.Num(); ++Next)
        {
            if (Evidence[Next].Type == EOffgridAIAudioLandmarkType::Resume
                && Evidence[Next].DecisionSec <= Input.ObservedAudioBufferEndSec + 0.001f)
            {
                Lull.EndSeconds = Evidence[Next].CenterSec;
                break;
            }
        }
        InOutTrack.AcousticLulls.Add(Lull);
    }

    float LastCenter = LastCommittedCenter(InOutTrack);
    for (const auto& Beat : Evidence)
    {
        if (Beat.Type != EOffgridAIAudioLandmarkType::SyllabicPulse
            || (!Input.bInputStreamClosed
                && Beat.DecisionSec + AudioPulseCommitStabilitySec
                    > Input.ObservedAudioBufferEndSec + 0.001f)
            || Beat.CenterSec <= State.LastMatchedSyllableAudioSec + 0.001f)
            continue;

        // The valley is an explicit full lip closure so the diagnostic beat is
        // unmistakable in video. It carries no phonetic identity.
        if (State.LastMatchedSyllableAudioSec >= 0.0f)
        {
            const float ValleySec = FindInterNucleusValley(
                Input,
                State.LastMatchedSyllableAudioSec,
                Beat.CenterSec);
            if (ValleySec > LastCenter + 0.001f
                && ValleySec < Beat.CenterSec - 0.001f)
            {
                FOffgridAICommittedVisemeEvent CloseEvent;
                CloseEvent.EventIndex = InOutTrack.Events.Num();
                CloseEvent.PoseID = FName(TEXT("22_MBP"));
                CloseEvent.Strength = 1.0f;
                CloseEvent.SpeechRegionIndex = RegionContaining(
                    *Input.SpeechRegions, ValleySec);
                CloseEvent.bIsStrongVisibleEvent = true;
                CloseEvent.bIsRenderable = true;
                CloseEvent.FinalRenderCenterSeconds = ValleySec;
                CloseEvent.RenderStartSeconds = FMath::Max(ValleySec - 0.030f, 0.0f);
                CloseEvent.RenderEndSeconds = ValleySec + 0.030f;
                CloseEvent.PriorStartSeconds = CloseEvent.RenderStartSeconds;
                CloseEvent.PriorCenterSeconds = ValleySec;
                CloseEvent.PriorEndSeconds = CloseEvent.RenderEndSeconds;
                CloseEvent.LeadAdjustedCenterSeconds = ValleySec;
                CloseEvent.SourcePhoneClass = FName(TEXT("nucleus_gap"));
                CloseEvent.bMappedToObservedSpeech =
                    CloseEvent.SpeechRegionIndex != INDEX_NONE;
                CloseEvent.CommitPlaybackSeconds = Input.CurrentPlaybackSec;
                CloseEvent.CommitLeadSeconds = ValleySec - Input.CurrentPlaybackSec;
                CloseEvent.CommitReason = FName(TEXT("nucleus_valley_close"));
                CloseEvent.AcousticAnchorKind = FName(TEXT("inter_nucleus_valley"));
                CloseEvent.AcousticAnchorSeconds = ValleySec;
                CloseEvent.AcousticAnchorErrorSeconds = 0.0f;
                InOutTrack.Events.Add(CloseEvent);
                LastCenter = ValleySec;
            }
        }

        FOffgridAICommittedVisemeEvent Event;
        Event.EventIndex = InOutTrack.Events.Num();
        Event.PoseID = FName(TEXT("08_Ah"));
        Event.Strength = 1.0f;
        Event.SpeechRegionIndex = RegionContaining(
            *Input.SpeechRegions, Beat.CenterSec);
        Event.bIsStrongVisibleEvent = true;
        Event.bIsRenderable = true;
        Event.FinalRenderCenterSeconds = Beat.CenterSec;
        Event.RenderStartSeconds = FMath::Max(Beat.CenterSec - 0.035f, 0.0f);
        Event.RenderEndSeconds = Beat.CenterSec + 0.035f;
        Event.PriorStartSeconds = Event.RenderStartSeconds;
        Event.PriorCenterSeconds = Beat.CenterSec;
        Event.PriorEndSeconds = Event.RenderEndSeconds;
        Event.LeadAdjustedCenterSeconds = Beat.CenterSec;
        Event.SourcePhoneClass = FName(TEXT("acoustic_nucleus_beat"));
        Event.bMappedToObservedSpeech = Event.SpeechRegionIndex != INDEX_NONE;
        Event.CommitPlaybackSeconds = Input.CurrentPlaybackSec;
        Event.CommitLeadSeconds = Beat.CenterSec - Input.CurrentPlaybackSec;
        Event.CommitReason = FName(TEXT("acoustic_nucleus_open"));
        Event.AcousticAnchorKind = FName(TEXT("acoustic_nucleus_pulse"));
        Event.AcousticAnchorSeconds = Beat.CenterSec;
        Event.AcousticAnchorErrorSeconds = 0.0f;
        InOutTrack.Events.Add(Event);
        LastCenter = Beat.CenterSec;
        ++State.LastMatchedSyllableIndex;
        State.LastMatchedSyllablePhoneIndex = INDEX_NONE;
        State.LastMatchedSyllableAudioSec = Beat.CenterSec;
        State.LastMatchedSyllableConfidence = Beat.Score;
    }

    if (Input.SpeechRegions->Num() > 0)
    {
        InOutTrack.SpeechStartSeconds = (*Input.SpeechRegions)[0].AudioBufferStartSec;
        InOutTrack.SpeechEndSeconds = Input.SpeechRegions->Last().AudioBufferEndSec;
    }
    State.NextTextEventIndex = FMath::Max(State.LastMatchedSyllableIndex + 1, 0);
    State.SchedulerNextEventIndex = State.NextTextEventIndex;
    State.SchedulerBlockReason = FName(TEXT("nucleus_beat_experiment"));
    State.bPlayheadStarted = Input.SpeechRegions->Num() > 0;
    bInOutTrackBuilt = true;
}

static void UpdateSyllablePacedVisemeTrack(
    const FOffgridAILipsyncRuntimeUpdateInput& Input,
    FOffgridAICommittedVisemeTrack& InOutTrack,
    FOffgridAIBoundaryPlaybackState& State,
    bool& bInOutTrackBuilt)
{
    if (!Input.TextPlan || !Input.AudioFeatureFrames || !Input.SpeechRegions) return;
    const FOffgridAITextVisemePlan& Plan = *Input.TextPlan;
    const FRuntimePrior Prior = BuildRuntimePrior(Plan);

    InOutTrack.NPCID = Input.NPCID;
    InOutTrack.LineID = Input.LineID;
    InOutTrack.bSyllablePacedVisemesExperiment = true;
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
    Config.PostrollSec = 0.250f;
    Config.bPermissivePhoneCandidates = true;
    Config.bNucleusBeatIndicatorTuning = true;
    Config.SpeechRegions = Input.SpeechRegions;
    const TArray<FOffgridAIAudioLandmarkObservation> Evidence =
        FOffgridAIStreamingEvidenceSurface::Analyze(*Input.AudioFeatureFrames, Config);
    const TArray<FOffgridAIStreamingSyllableCandidateSet> CandidateSets =
        FOffgridAIStreamingSyllablePositionEstimator::EstimateCandidateSets(
            Plan, Evidence, 4, 6, 4);

    float LastCenter = LastCommittedCenter(InOutTrack);
    for (const auto& CandidateSet : CandidateSets)
    {
        if (CandidateSet.AudioCenterSec <= State.LastProcessedSyllablePulseSec + 0.001f
            || (!Input.bInputStreamClosed
                && CandidateSet.DecisionSec + AudioPulseCommitStabilitySec
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
        bool bLaterPulseInRegion = false;
        for (const auto& LaterSet : CandidateSets)
        {
            if (LaterSet.AudioCenterSec <= CandidateSet.AudioCenterSec + 0.001f)
                continue;
            if (RegionContaining(*Input.SpeechRegions, LaterSet.AudioCenterSec)
                == AudioRegionIndex)
            {
                bLaterPulseInRegion = true;
                break;
            }
        }
        const bool bRegionCloseSupported =
            Input.SpeechRegions->IsValidIndex(AudioRegionIndex)
            && ((*Input.SpeechRegions)[AudioRegionIndex].bEnded
                || ((*Input.SpeechRegions)[AudioRegionIndex].ProvisionalEndSec >= 0.0f
                    && Input.ObservedAudioBufferEndSec
                        - (*Input.SpeechRegions)[AudioRegionIndex].ProvisionalEndSec
                        >= 0.100f));
        const bool bFinalPulseInClosedRegion =
            bRegionCloseSupported && !bLaterPulseInRegion;
        for (int32 CandidateIndex = 0;
            CandidateIndex < CandidateSet.SyllableIndices.Num();
            ++CandidateIndex)
        {
            const int32 SyllableIndex = CandidateSet.SyllableIndices[CandidateIndex];
            if (AssignedSyllableIndex != INDEX_NONE
                && SyllableIndex != AssignedSyllableIndex)
                continue;
            if (SyllableIndex <= State.LastMatchedSyllableIndex
                || !Plan.Syllables.IsValidIndex(SyllableIndex)
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
                    || !Plan.Syllables.IsValidIndex(SyllableIndex))
                    continue;
                AssignedSyllableIndex = SyllableIndex;
                AssignmentScore = CandidateSet.Scores.IsValidIndex(CandidateIndex)
                    ? CandidateSet.Scores[CandidateIndex]
                    : 0.0f;
                break;
            }
        }
        if (bFinalPulseInClosedRegion && AssignedSyllableIndex != INDEX_NONE)
        {
            int32 LastSyllableInPlannedRegion = INDEX_NONE;
            for (int32 SyllableIndex = Plan.Syllables.Num() - 1;
                SyllableIndex > State.LastMatchedSyllableIndex;
                --SyllableIndex)
            {
                if (Plan.Syllables[SyllableIndex].SpeechRegionIndex
                    == Plan.Syllables[AssignedSyllableIndex].SpeechRegionIndex)
                {
                    LastSyllableInPlannedRegion = SyllableIndex;
                    break;
                }
            }
            if (LastSyllableInPlannedRegion != INDEX_NONE
                && LastSyllableInPlannedRegion - State.LastMatchedSyllableIndex <= 2)
            {
                AssignedSyllableIndex = LastSyllableInPlannedRegion;
            }
            else
            {
                const int32 CandidateWordIndex =
                    Plan.Syllables[AssignedSyllableIndex].WordIndex;
                while (Plan.Syllables.IsValidIndex(AssignedSyllableIndex + 1)
                    && Plan.Syllables[AssignedSyllableIndex + 1].WordIndex
                        == CandidateWordIndex)
                {
                    ++AssignedSyllableIndex;
                }
            }
            AssignmentScore = 0.0f;
            for (int32 CandidateIndex = 0;
                CandidateIndex < CandidateSet.SyllableIndices.Num();
                ++CandidateIndex)
            {
                if (CandidateSet.SyllableIndices[CandidateIndex]
                    == AssignedSyllableIndex)
                {
                    AssignmentScore = CandidateSet.Scores.IsValidIndex(CandidateIndex)
                        ? CandidateSet.Scores[CandidateIndex]
                        : 0.0f;
                    break;
                }
            }
        }
        // The final region pulse may need to absorb a detector false negative.
        // Its target can fall outside the beam's short candidate list; retain
        // the region-close assignment with zero confidence so the recovery is
        // visible in diagnostics rather than leaking correspondence forward.
        if (bFinalPulseInClosedRegion && AssignedSyllableIndex != INDEX_NONE
            && AssignmentScore == 0.0f)
        {
            AssignmentScore = 0.45f;
        }
        if (AssignedSyllableIndex == INDEX_NONE) continue;

        int32 LastPlannedSyllableInRegion = INDEX_NONE;
        for (int32 SyllableIndex = Plan.Syllables.Num() - 1;
            SyllableIndex >= 0;
            --SyllableIndex)
        {
            if (Plan.Syllables[SyllableIndex].SpeechRegionIndex == AudioRegionIndex)
            {
                LastPlannedSyllableInRegion = SyllableIndex;
                break;
            }
        }
        const bool bWouldEnterFinalRegionPair =
            LastPlannedSyllableInRegion != INDEX_NONE
            && Plan.Syllables[AssignedSyllableIndex].SpeechRegionIndex
                == AudioRegionIndex
            && LastPlannedSyllableInRegion - AssignedSyllableIndex <= 1;
        const bool bWouldSplitCurrentWord =
            Plan.Syllables.IsValidIndex(AssignedSyllableIndex + 1)
            && Plan.Syllables[AssignedSyllableIndex + 1].WordIndex
                == Plan.Syllables[AssignedSyllableIndex].WordIndex;
        const bool bAlreadyStartedRegion =
            State.LastMatchedSyllableIndex >= 0
            && Plan.Syllables.IsValidIndex(State.LastMatchedSyllableIndex)
            && Plan.Syllables[State.LastMatchedSyllableIndex].SpeechRegionIndex
                == AudioRegionIndex;
        if (!bRegionCloseSupported
            && !bLaterPulseInRegion
            && (bWouldEnterFinalRegionPair || bWouldSplitCurrentWord)
            && bAlreadyStartedRegion)
        {
            // Hold the possible region-final pulse briefly. If the region
            // closes it reconciles any remaining syllable count locally; if
            // speech continues, a following pulse proves it was not final.
            continue;
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
                return CandidateSet.AudioCenterSec;
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
                    State,
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
        && State.LastWordAnchorAudioSec >= 0.0f)
    {
        const int32 AudioRegionIndex = Input.SpeechRegions->Num() > 0
            ? Input.SpeechRegions->Num() - 1
            : INDEX_NONE;
        const float FixedWordRate = FMath::Clamp(
            State.AdaptiveWordPriorRate,
            MinimumAdaptiveWordPriorRate,
            MaximumAdaptiveWordPriorRate);
        float LastTailRecoveryAnchorSec = -1.0f;
        for (int32 WordIndex = State.LastCommittedWordIndex + 1;
            WordIndex < Plan.WordPhoneBeginIndices.Num();
            ++WordIndex)
        {
            const int32 AnchorEventIndex =
                FirstPerceptuallyRenderedEventForWord(Plan, WordIndex);
            if (!Prior.EventCenters.IsValidIndex(AnchorEventIndex)) continue;
            const float PriorAnchorSec = Prior.EventCenters[AnchorEventIndex];
            const float ProjectedAnchorSec = State.LastWordAnchorAudioSec
                + (PriorAnchorSec - State.LastWordAnchorPriorSec) / FixedWordRate;
            const float WordAnchorSec = FMath::Max(
                FMath::Max(
                    ProjectedAnchorSec,
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
                    Input, Plan, Prior, State, EventIndex, AudioRegionIndex,
                    CenterSec,
                    FName(TEXT("audio_word_prior_stream_tail_recovery")),
                    Event);
                Event.AcousticAnchorKind = FName(TEXT("projected_word_start"));
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
            Row.AnchorKind = FName(TEXT("audio_word_start_prior_stream_tail_recovery"));
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

void FOffgridAILipsyncRuntimeAdapter::UpdateCommittedTrack(
    const FOffgridAILipsyncRuntimeUpdateInput& Input,
    FOffgridAICommittedVisemeTrack& InOutTrack,
    FOffgridAIBoundaryPlaybackState& InOutState,
    bool& bInOutTrackBuilt)
{
    if (Input.bEnableAudioPulseMouthExperiment)
    {
        UpdateAudioPulseMouthTrack(Input, InOutTrack, InOutState, bInOutTrackBuilt);
        return;
    }
    if (Input.bEnableSyllablePacedVisemesExperiment)
    {
        UpdateSyllablePacedVisemeTrack(Input, InOutTrack, InOutState, bInOutTrackBuilt);
        return;
    }
    UpdateSimpleCommittedTrack(Input, InOutTrack, InOutState, bInOutTrackBuilt);
}
