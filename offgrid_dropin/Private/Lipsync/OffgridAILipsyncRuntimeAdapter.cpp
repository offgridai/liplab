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
    Detector.SetListGapSensitivity(
        bEnableFocusedWordStartAlignment
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

void FOffgridAILipsyncRuntimeAdapter::UpdateCommittedTrack(
    const FOffgridAILipsyncRuntimeUpdateInput& Input,
    FOffgridAICommittedVisemeTrack& InOutTrack,
    FOffgridAIBoundaryPlaybackState& InOutState,
    bool& bInOutTrackBuilt)
{
    UpdateSimpleCommittedTrack(Input, InOutTrack, InOutState, bInOutTrackBuilt);
}
