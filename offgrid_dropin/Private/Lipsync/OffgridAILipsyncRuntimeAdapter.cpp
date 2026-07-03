#include "Lipsync/OffgridAILipsyncRuntimeAdapter.h"

#include "Lipsync/OffgridAITextVisemePlanner.h"
#include "Lipsync/OffgridAIOnlinePhoneAligner.h"

namespace
{
constexpr float StreamingCommitLagSec = 0.100f;

static float EventCenterNorm(const FOffgridAITextVisemeEvent& E)
{
    return FMath::Clamp((E.StartNorm + E.EndNorm) * 0.5f, 0.0f, 1.0f);
}

static FName ResolvePose(const FOffgridAITextVisemeEvent& E)
{
    return E.PoseID.IsNone() ? FName(FOffgridAITextVisemePlanner::ToPoseKey(E.Viseme)) : E.PoseID;
}

static bool IsPose(FName PoseID, const TCHAR* Literal)
{
    return PoseID == FName(Literal);
}

static bool IsVowelPose(FName PoseID)
{
    return IsPose(PoseID, TEXT("07_Aa")) || IsPose(PoseID, TEXT("06_Eh")) || IsPose(PoseID, TEXT("03_Ee"))
        || IsPose(PoseID, TEXT("09_Oh")) || IsPose(PoseID, TEXT("11_Oo")) || IsPose(PoseID, TEXT("12_Ww-Oo-"));
}

static bool IsStrongPose(FName PoseID)
{
    return IsPose(PoseID, TEXT("22_MBP")) || IsPose(PoseID, TEXT("20_FV")) || IsPose(PoseID, TEXT("14_ChJjSh"))
        || IsPose(PoseID, TEXT("12_Ww-Oo-"));
}

static float SpanForPose(FName PoseID)
{
    if (IsPose(PoseID, TEXT("22_MBP"))) return 0.105f;
    if (IsPose(PoseID, TEXT("20_FV"))) return 0.100f;
    if (IsPose(PoseID, TEXT("14_ChJjSh"))) return 0.095f;
    if (IsVowelPose(PoseID)) return 0.135f;
    return 0.110f;
}

static float LeadForPose(FName PoseID)
{
    if (IsPose(PoseID, TEXT("22_MBP"))) return 0.035f;
    if (IsPose(PoseID, TEXT("20_FV"))) return 0.025f;
    if (IsPose(PoseID, TEXT("14_ChJjSh"))) return 0.018f;
    return 0.006f;
}

static float CenterForAlignedPhone(FName PoseID, EOffgridAIPhoneClass PhoneClass, float PhoneStart, float PhoneEnd)
{
    const float Duration = FMath::Max(PhoneEnd - PhoneStart, 0.0f);
    if (Duration <= 0.0f)
    {
        return PhoneStart;
    }

    switch (PhoneClass)
    {
    case EOffgridAIPhoneClass::Bilabial:
        return PhoneStart + FMath::Min(0.018f, Duration * 0.35f);
    case EOffgridAIPhoneClass::Labiodental:
        return PhoneStart + FMath::Min(0.022f, Duration * 0.40f);
    case EOffgridAIPhoneClass::Dental:
    case EOffgridAIPhoneClass::Sibilant:
    case EOffgridAIPhoneClass::StopBurst:
        return PhoneStart + FMath::Min(0.020f, Duration * 0.35f);
    case EOffgridAIPhoneClass::Liquid:
    case EOffgridAIPhoneClass::Glide:
    case EOffgridAIPhoneClass::Nasal:
        return PhoneStart + Duration * 0.38f;
    case EOffgridAIPhoneClass::VowelOpen:
    case EOffgridAIPhoneClass::VowelFront:
    case EOffgridAIPhoneClass::VowelRound:
        return PhoneStart + Duration * 0.42f;
    default:
        return ((PhoneStart + PhoneEnd) * 0.5f) - LeadForPose(PoseID);
    }
}

static float CommitConfidenceThresholdForClass(EOffgridAIPhoneClass PhoneClass, FName PoseID)
{
    switch (PhoneClass)
    {
    case EOffgridAIPhoneClass::Bilabial: return 0.22f;
    case EOffgridAIPhoneClass::Labiodental: return 0.24f;
    case EOffgridAIPhoneClass::Dental: return 0.26f;
    case EOffgridAIPhoneClass::Sibilant: return 0.28f;
    case EOffgridAIPhoneClass::StopBurst: return 0.20f;
    case EOffgridAIPhoneClass::Liquid:
    case EOffgridAIPhoneClass::Glide:
    case EOffgridAIPhoneClass::Nasal: return 0.22f;
    case EOffgridAIPhoneClass::VowelOpen:
    case EOffgridAIPhoneClass::VowelFront:
    case EOffgridAIPhoneClass::VowelRound: return 0.18f;
    default: return IsStrongPose(PoseID) ? 0.20f : 0.24f;
    }
}

static float StrengthForEvent(const FOffgridAITextVisemeEvent& E, FName PoseID)
{
    float S = FMath::Clamp(E.Strength, 0.18f, 1.0f);
    if (IsStrongPose(PoseID)) S = FMath::Max(S, 0.78f);
    else if (IsVowelPose(PoseID)) S = FMath::Max(S, 0.58f);
    return FMath::Clamp(S, 0.16f, 1.0f);
}

static int32 FirstUncommittedEventIndex(const FOffgridAIAlignedVisemeTrack& Track)
{
    int32 Next = 0;
    for (const FOffgridAIAlignedVisemeEvent& E : Track.Events)
    {
        Next = FMath::Max(Next, E.EventIndex + 1);
    }
    return Next;
}

static float LastCommittedCenter(const FOffgridAIAlignedVisemeTrack& Track)
{
    return Track.Events.Num() > 0 ? Track.Events.Last().FinalRenderCenterSeconds : 0.0f;
}

static float FirstConfirmedSpeechStart(const TArray<FOffgridAIStreamingSpeechIsland>* Islands)
{
    if (!Islands || Islands->Num() == 0)
    {
        return -1.0f;
    }

    float First = TNumericLimits<float>::Max();
    for (const FOffgridAIStreamingSpeechIsland& Island : *Islands)
    {
        if (Island.AudioBufferEndSec > Island.AudioBufferStartSec)
        {
            First = FMath::Min(First, Island.AudioBufferStartSec);
        }
    }
    return First == TNumericLimits<float>::Max() ? -1.0f : First;
}

static bool SameSentenceAsLast(const FOffgridAIAlignedVisemeTrack& Track, const FOffgridAITextVisemeEvent& T)
{
    if (Track.Events.Num() == 0)
    {
        return false;
    }
    const FOffgridAIAlignedVisemeEvent& Last = Track.Events.Last();
    return Last.SentenceIndex != INDEX_NONE && T.SentenceIslandIndex != INDEX_NONE && Last.SentenceIndex == T.SentenceIslandIndex;
}

static float MaxWallClockGapForPose(FName PoseID)
{
    // Research intent: this is an HMM-style state-exit prior, not a post-hoc
    // island scheduler. During one text sentence, a visible phone state may not
    // occupy an arbitrarily long wall-clock interval just because the acoustic
    // evidence is weak or sparse.
    if (IsPose(PoseID, TEXT("22_MBP")) || IsPose(PoseID, TEXT("20_FV")) || IsPose(PoseID, TEXT("14_ChJjSh")))
    {
        return 0.220f;
    }
    if (IsVowelPose(PoseID))
    {
        return 0.300f;
    }
    return 0.260f;
}


static TCHAR BoundaryPunctuationBeforeEvent(const FOffgridAITextVisemePlan& Plan, int32 EventIndex)
{
    if (!Plan.Events.IsValidIndex(EventIndex))
    {
        return TCHAR(0);
    }

    const int32 WordIndex = Plan.Events[EventIndex].WordIndex;
    const int32 BoundaryWordIndex = WordIndex - 1;
    return Plan.WordBoundaryPunctuationAfter.IsValidIndex(BoundaryWordIndex)
        ? Plan.WordBoundaryPunctuationAfter[BoundaryWordIndex]
        : TCHAR(0);
}

static bool IsSoftCommaBoundary(TCHAR Punctuation)
{
    return Punctuation == TCHAR(',');
}

static float RuntimeSpeechEvidenceScore(const FOffgridAIStreamingAudioFeatureFrame& F);
static bool RuntimeLowEvidenceFrame(const FOffgridAIStreamingAudioFeatureFrame& F);

static bool IsStrongSentenceBoundaryPunctuation(TCHAR Punctuation)
{
    return Punctuation == TCHAR('.')
        || Punctuation == TCHAR('!')
        || Punctuation == TCHAR('?');
}

static float ClampedIslandEnd(const FOffgridAIStreamingSpeechIsland& Island, float ObservedAudioEndSec)
{
    float End = FMath::Max(Island.AudioBufferLastSpeechSec, Island.AudioBufferEndSec);
    if (ObservedAudioEndSec > 0.0f)
    {
        End = FMath::Min(End, ObservedAudioEndSec);
    }
    return End;
}

static TArray<float> BuildSentenceWeights(const FOffgridAITextVisemePlan& Plan, int32 SentenceCount)
{
    TArray<float> Weights;
    Weights.Init(0.0f, SentenceCount);
    for (const FOffgridAIExpectedPhone& Phone : Plan.ExpectedPhones)
    {
        if (Weights.IsValidIndex(Phone.SentenceIslandIndex))
        {
            Weights[Phone.SentenceIslandIndex] += FMath::Max(Phone.WeightSeconds, 0.030f);
        }
    }
    for (int32 WordIndex = 0; WordIndex < Plan.WordSentenceIslandIndices.Num(); ++WordIndex)
    {
        const int32 SentenceIndex = Plan.WordSentenceIslandIndices[WordIndex];
        if (Weights.IsValidIndex(SentenceIndex) && Weights[SentenceIndex] <= 0.0f)
        {
            Weights[SentenceIndex] += 0.080f;
        }
    }
    for (float& Weight : Weights)
    {
        Weight = FMath::Max(Weight, 0.080f);
    }
    return Weights;
}

static TCHAR SentenceBoundaryPunctuationAfterSentence(const FOffgridAITextVisemePlan& Plan, int32 SentenceIndex)
{
    int32 BoundaryWordIndex = INDEX_NONE;
    for (int32 WordIndex = 0; WordIndex < Plan.WordSentenceIslandIndices.Num(); ++WordIndex)
    {
        if (Plan.WordSentenceIslandIndices[WordIndex] == SentenceIndex)
        {
            BoundaryWordIndex = WordIndex;
        }
    }
    return Plan.WordBoundaryPunctuationAfter.IsValidIndex(BoundaryWordIndex)
        ? Plan.WordBoundaryPunctuationAfter[BoundaryWordIndex]
        : TCHAR(0);
}

static FOffgridAIStreamingSpeechIsland MergeSpeechIslands(
    const FOffgridAIStreamingSpeechIsland& A,
    const FOffgridAIStreamingSpeechIsland& B)
{
    FOffgridAIStreamingSpeechIsland Merged = A;
    Merged.AudioBufferLastSpeechSec = FMath::Max(A.AudioBufferLastSpeechSec, B.AudioBufferLastSpeechSec);
    Merged.AudioBufferEndSec = FMath::Max(A.AudioBufferEndSec, B.AudioBufferEndSec);
    Merged.bStarted = A.bStarted || B.bStarted;
    Merged.bEnded = A.bEnded || B.bEnded;
    Merged.ProvisionalEndSec = -1.0f;
    Merged.EndDecisionSec = B.EndDecisionSec > 0.0f ? B.EndDecisionSec : A.EndDecisionSec;
    Merged.ReopenCount = A.ReopenCount + B.ReopenCount;
    Merged.EndReason = FName(TEXT("resolved_gap_merge"));
    return Merged;
}

static void RenumberSpeechIslands(TArray<FOffgridAIStreamingSpeechIsland>& Islands)
{
    for (int32 Index = 0; Index < Islands.Num(); ++Index)
    {
        Islands[Index].IslandIndex = Index;
    }
}

static TArray<float> DetectSentenceBoundarySplits(
    const FOffgridAITextVisemePlan& Plan,
    const TArray<FOffgridAIStreamingSpeechIsland>& RawIslands,
    const TArray<FOffgridAIStreamingAudioFeatureFrame>& Frames,
    float ObservedAudioEndSec)
{
    TArray<float> Splits;
    if (RawIslands.Num() == 0 || Frames.Num() == 0)
    {
        return Splits;
    }

    int32 SentenceCount = 0;
    for (int32 SentenceIndex : Plan.WordSentenceIslandIndices)
    {
        SentenceCount = FMath::Max(SentenceCount, SentenceIndex + 1);
    }
    if (SentenceCount <= 1)
    {
        return Splits;
    }

    const TArray<float> SentenceWeights = BuildSentenceWeights(Plan, SentenceCount);
    float TotalWeight = 0.0f;
    for (float Weight : SentenceWeights)
    {
        TotalWeight += Weight;
    }
    if (TotalWeight <= 0.0f)
    {
        return Splits;
    }

    const float OverallStart = RawIslands[0].AudioBufferStartSec;
    const float OverallEnd = ClampedIslandEnd(RawIslands.Last(), ObservedAudioEndSec);
    const float OverallDuration = OverallEnd - OverallStart;
    if (OverallDuration <= 0.20f)
    {
        return Splits;
    }

    float CumulativeWeight = 0.0f;
    for (int32 SentenceIndex = 0; SentenceIndex < SentenceCount - 1; ++SentenceIndex)
    {
        const TCHAR BoundaryPunctuation = SentenceBoundaryPunctuationAfterSentence(Plan, SentenceIndex);
        CumulativeWeight += SentenceWeights[SentenceIndex];
        const float BoundaryProgress = FMath::Clamp(CumulativeWeight / TotalWeight, 0.0f, 1.0f);
        const float TargetTimeSec = OverallStart + BoundaryProgress * OverallDuration;
        const float SearchRadiusSec = IsStrongSentenceBoundaryPunctuation(BoundaryPunctuation) ? 0.34f : 0.22f;
        const float SearchStartSec = FMath::Max(OverallStart + 0.08f, TargetTimeSec - SearchRadiusSec);
        const float SearchEndSec = FMath::Min(OverallEnd - 0.08f, TargetTimeSec + SearchRadiusSec);
        if (SearchEndSec <= SearchStartSec)
        {
            continue;
        }

        float BestScore = -1.0f;
        float BestTimeSec = -1.0f;
        float BestEvidence = 1.0f;
        for (const FOffgridAIStreamingAudioFeatureFrame& Frame : Frames)
        {
            if (Frame.AudioBufferCenterSec < SearchStartSec || Frame.AudioBufferCenterSec > SearchEndSec)
            {
                continue;
            }

            const float Evidence = RuntimeSpeechEvidenceScore(Frame);
            const float DistancePenalty = FMath::Abs(Frame.AudioBufferCenterSec - TargetTimeSec) / FMath::Max(SearchRadiusSec, 0.001f);
            float Score = (1.0f - Evidence) * 0.62f;
            Score += RuntimeLowEvidenceFrame(Frame) ? 0.20f : 0.0f;
            Score += Frame.bLocalRMSValley ? 0.10f : 0.0f;
            Score += (Frame.Periodicity < 0.22f && Frame.Flux < 0.06f) ? 0.08f : 0.0f;
            Score -= FMath::Clamp(DistancePenalty, 0.0f, 1.0f) * 0.12f;

            if (Score > BestScore)
            {
                BestScore = Score;
                BestTimeSec = Frame.AudioBufferCenterSec;
                BestEvidence = Evidence;
            }
        }

        if (BestTimeSec <= 0.0f)
        {
            continue;
        }

        const bool bAcceptStrongBoundary = BestScore >= 0.60f && BestEvidence <= 0.14f;
        const bool bAcceptWeakBoundary = BestScore >= 0.72f && BestEvidence <= 0.11f;
        if ((IsStrongSentenceBoundaryPunctuation(BoundaryPunctuation) && bAcceptStrongBoundary)
            || (!IsStrongSentenceBoundaryPunctuation(BoundaryPunctuation) && bAcceptWeakBoundary))
        {
            if (Splits.Num() == 0 || FMath::Abs(Splits.Last() - BestTimeSec) >= 0.12f)
            {
                Splits.Add(BestTimeSec);
            }
        }
    }

    return Splits;
}

static TArray<FOffgridAIStreamingSpeechIsland> ResolveSpeechIslands(
    const FOffgridAITextVisemePlan& Plan,
    const TArray<FOffgridAIStreamingSpeechIsland>& DetectorIslands,
    const TArray<FOffgridAIStreamingAudioFeatureFrame>& Frames,
    float ObservedAudioEndSec)
{
    TArray<FOffgridAIStreamingSpeechIsland> Raw;
    for (const FOffgridAIStreamingSpeechIsland& Island : DetectorIslands)
    {
        const float End = ClampedIslandEnd(Island, ObservedAudioEndSec);
        if (End <= Island.AudioBufferStartSec + 0.040f)
        {
            continue;
        }
        FOffgridAIStreamingSpeechIsland Copy = Island;
        Copy.AudioBufferEndSec = End;
        Copy.AudioBufferLastSpeechSec = FMath::Min(FMath::Max(Island.AudioBufferLastSpeechSec, Island.AudioBufferStartSec), End);
        Raw.Add(Copy);
    }
    if (Raw.Num() <= 1)
    {
        RenumberSpeechIslands(Raw);
        return Raw;
    }

    const TArray<float> SentenceSplits = DetectSentenceBoundarySplits(Plan, Raw, Frames, ObservedAudioEndSec);

    TArray<FOffgridAIStreamingSpeechIsland> Merged;
    Merged.Add(Raw[0]);
    for (int32 Index = 1; Index < Raw.Num(); ++Index)
    {
        const FOffgridAIStreamingSpeechIsland& Next = Raw[Index];
        const FOffgridAIStreamingSpeechIsland& Prev = Merged.Last();
        const float GapStartSec = ClampedIslandEnd(Prev, ObservedAudioEndSec);
        const float GapEndSec = Next.AudioBufferStartSec;
        const float GapDurationSec = FMath::Max(GapEndSec - GapStartSec, 0.0f);

        bool bKeepSplit = GapDurationSec >= 0.180f;
        if (!bKeepSplit)
        {
            for (float SplitSec : SentenceSplits)
            {
                if (SplitSec >= GapStartSec - 0.025f && SplitSec <= GapEndSec + 0.025f)
                {
                    bKeepSplit = true;
                    break;
                }
            }
        }
        if (!bKeepSplit)
        {
            const bool bStrongClose = Prev.EndReason == FName(TEXT("strong_quiet_hangover"))
                || Prev.EndReason == FName(TEXT("finalize_at_provisional_end"));
            bKeepSplit = bStrongClose && GapDurationSec >= 0.105f;
        }

        if (bKeepSplit)
        {
            Merged.Add(Next);
        }
        else
        {
            Merged.Last() = MergeSpeechIslands(Prev, Next);
        }
    }

    TArray<FOffgridAIStreamingSpeechIsland> Resolved;
    for (const FOffgridAIStreamingSpeechIsland& Island : Merged)
    {
        TArray<float> IslandSplits;
        for (float SplitSec : SentenceSplits)
        {
            if (SplitSec > Island.AudioBufferStartSec + 0.08f && SplitSec < Island.AudioBufferEndSec - 0.08f)
            {
                IslandSplits.Add(SplitSec);
            }
        }

        if (IslandSplits.Num() == 0)
        {
            Resolved.Add(Island);
            continue;
        }

        float SegmentStartSec = Island.AudioBufferStartSec;
        for (float SplitSec : IslandSplits)
        {
            FOffgridAIStreamingSpeechIsland Segment = Island;
            Segment.AudioBufferStartSec = SegmentStartSec;
            Segment.AudioBufferLastSpeechSec = SplitSec;
            Segment.AudioBufferEndSec = SplitSec;
            Segment.ProvisionalEndSec = SplitSec;
            Segment.EndDecisionSec = SplitSec;
            Segment.EndReason = FName(TEXT("resolved_sentence_boundary_split"));
            Segment.ReopenCount = 0;
            Segment.bStarted = true;
            Segment.bEnded = true;
            Resolved.Add(Segment);
            SegmentStartSec = SplitSec;
        }

        FOffgridAIStreamingSpeechIsland Tail = Island;
        Tail.AudioBufferStartSec = SegmentStartSec;
        Tail.bStarted = true;
        Tail.bEnded = true;
        Resolved.Add(Tail);
    }

    RenumberSpeechIslands(Resolved);
    return Resolved;
}

static bool FindNextSpeechIslandStartAfter(const TArray<FOffgridAIStreamingSpeechIsland>* Islands, float AfterSec, float MinGapSec, float& OutStartSec)
{
    if (!Islands)
    {
        return false;
    }

    float Best = TNumericLimits<float>::Max();
    for (const FOffgridAIStreamingSpeechIsland& Island : *Islands)
    {
        const float Start = Island.AudioBufferStartSec;
        const float End = FMath::Max(Island.AudioBufferLastSpeechSec, Island.AudioBufferEndSec);
        if (End <= Start)
        {
            continue;
        }
        if (Start > AfterSec + MinGapSec && Start < Best)
        {
            Best = Start;
        }
    }

    if (Best == TNumericLimits<float>::Max())
    {
        return false;
    }
    OutStartSec = Best;
    return true;
}


static float RuntimeSpeechEvidenceScore(const FOffgridAIStreamingAudioFeatureFrame& F)
{
    return FMath::Clamp(FMath::Max(F.RMSNorm, FMath::Max(F.Flux * 0.55f, F.Periodicity * 0.45f)), 0.0f, 1.0f);
}

static bool RuntimeSpeechFrame(const FOffgridAIStreamingAudioFeatureFrame& F)
{
    return RuntimeSpeechEvidenceScore(F) >= 0.055f;
}

static bool RuntimeLowEvidenceFrame(const FOffgridAIStreamingAudioFeatureFrame& F)
{
    // This is deliberately stricter than the detector island close threshold.
    // Detector islands can bridge low-energy valleys; phrase anchoring needs to
    // see those valleys so comma/list boundaries do not leak into the prior word.
    return RuntimeSpeechEvidenceScore(F) < 0.040f && F.RMSNorm < 0.070f;
}


struct FOffgridAIMicroPauseTimingDiagnostics
{
    int32 Count50ms = 0;
    int32 Count75ms = 0;
    int32 Count120ms = 0;
    float NearestCenterSec = 0.0f;
    float NearestDurationSec = 0.0f;
    float NearestProgress01 = 0.0f;
};

static bool IsMicroPauseLowEvidenceFrame(const FOffgridAIStreamingAudioFeatureFrame& F)
{
    const FOffgridAIArticulatoryProbabilityField Field = FOffgridAIOnlinePhoneAligner::BuildArticulatoryProbabilityField(F);
    // Slightly looser than RuntimeLowEvidenceFrame: we are not splitting runtime
    // islands here, only asking whether short streaming-detectable valleys could
    // become useful timing anchors.  A frame qualifies when both raw energy and
    // the unified articulation field agree that speech is weak.
    return (RuntimeSpeechEvidenceScore(F) < 0.062f && F.RMSNorm < 0.085f && Field.Speech < 0.42f) || Field.Silence > 0.72f;
}

static FOffgridAIMicroPauseTimingDiagnostics AnalyzeMicroPausesInRegion(
    const TArray<FOffgridAIStreamingAudioFeatureFrame>* Frames,
    float RegionStartSec,
    float RegionEndSec,
    float CurrentPlaybackSec)
{
    FOffgridAIMicroPauseTimingDiagnostics Out;
    if (!Frames || Frames->Num() == 0 || RegionEndSec <= RegionStartSec)
    {
        return Out;
    }

    bool bInRun = false;
    float RunStart = 0.0f;
    float RunEnd = 0.0f;
    float BestDistance = TNumericLimits<float>::Max();

    auto CommitRun = [&](float Start, float End)
    {
        const float Duration = End - Start;
        if (Duration < 0.050f)
        {
            return;
        }
        if (Duration >= 0.050f) ++Out.Count50ms;
        if (Duration >= 0.075f) ++Out.Count75ms;
        if (Duration >= 0.120f) ++Out.Count120ms;

        const float Center = 0.5f * (Start + End);
        const float D = FMath::Abs(Center - CurrentPlaybackSec);
        if (D < BestDistance)
        {
            BestDistance = D;
            Out.NearestCenterSec = Center;
            Out.NearestDurationSec = Duration;
            Out.NearestProgress01 = FMath::Clamp((Center - RegionStartSec) / FMath::Max(RegionEndSec - RegionStartSec, 0.001f), 0.0f, 1.0f);
        }
    };

    for (const FOffgridAIStreamingAudioFeatureFrame& F : *Frames)
    {
        if (F.AudioBufferEndSec <= RegionStartSec)
        {
            continue;
        }
        if (F.AudioBufferStartSec >= RegionEndSec)
        {
            break;
        }
        // Ignore leading/trailing margin.  These diagnostics are about internal
        // micro-pauses that can subdivide an otherwise-valid speech region.
        if (F.AudioBufferCenterSec <= RegionStartSec + 0.035f || F.AudioBufferCenterSec >= RegionEndSec - 0.035f)
        {
            continue;
        }

        const bool bLow = IsMicroPauseLowEvidenceFrame(F);
        if (bLow)
        {
            if (!bInRun)
            {
                bInRun = true;
                RunStart = F.AudioBufferStartSec;
                RunEnd = F.AudioBufferEndSec;
            }
            else
            {
                RunEnd = F.AudioBufferEndSec;
            }
        }
        else if (bInRun)
        {
            CommitRun(RunStart, RunEnd);
            bInRun = false;
        }
    }
    if (bInRun)
    {
        CommitRun(RunStart, RunEnd);
    }
    return Out;
}

static bool FindNextFeatureSpeechOnsetAfter(
    const TArray<FOffgridAIStreamingAudioFeatureFrame>* Frames,
    float AfterSec,
    float MinSilentGapSec,
    float MaxSearchSec,
    float& OutGapStartSec,
    float& OutSpeechStartSec)
{
    if (!Frames || Frames->Num() == 0)
    {
        return false;
    }

    const float SearchEnd = AfterSec + FMath::Max(MaxSearchSec, MinSilentGapSec + 0.050f);
    bool bInLowRun = false;
    float LowRunStart = 0.0f;
    float LowAccum = 0.0f;
    bool bGapAccepted = false;
    float CandidateSpeechStart = -1.0f;
    float SpeechAccum = 0.0f;

    for (const FOffgridAIStreamingAudioFeatureFrame& F : *Frames)
    {
        if (F.AudioBufferEndSec <= AfterSec)
        {
            continue;
        }
        if (F.AudioBufferStartSec > SearchEnd)
        {
            break;
        }

        const float FrameDuration = FMath::Max(F.AudioBufferEndSec - F.AudioBufferStartSec, 0.0f);
        if (FrameDuration <= 0.0f)
        {
            continue;
        }

        if (!bGapAccepted)
        {
            if (RuntimeLowEvidenceFrame(F))
            {
                if (!bInLowRun)
                {
                    bInLowRun = true;
                    LowRunStart = F.AudioBufferStartSec;
                    LowAccum = 0.0f;
                }
                LowAccum += FrameDuration;
                if (LowAccum >= MinSilentGapSec)
                {
                    bGapAccepted = true;
                    OutGapStartSec = LowRunStart;
                }
            }
            else
            {
                bInLowRun = false;
                LowAccum = 0.0f;
            }
            continue;
        }

        if (RuntimeSpeechFrame(F))
        {
            if (CandidateSpeechStart < 0.0f)
            {
                CandidateSpeechStart = F.AudioBufferStartSec;
                SpeechAccum = 0.0f;
            }
            SpeechAccum += FrameDuration;
            if (SpeechAccum >= 0.025f)
            {
                OutSpeechStartSec = CandidateSpeechStart;
                return true;
            }
        }
        else if (!RuntimeLowEvidenceFrame(F))
        {
            // Weak ambiguous frames neither prove speech nor reset a confirmed gap.
            CandidateSpeechStart = -1.0f;
            SpeechAccum = 0.0f;
        }
    }

    return false;
}



static bool HasSustainedRuntimeSpeechBetween(
    const TArray<FOffgridAIStreamingAudioFeatureFrame>* Frames,
    float StartSec,
    float EndSec,
    float MinSpeechSec)
{
    if (!Frames || EndSec <= StartSec + 0.020f)
    {
        return false;
    }

    float SpeechAccum = 0.0f;
    float LongestRun = 0.0f;
    float CurrentRun = 0.0f;
    for (const FOffgridAIStreamingAudioFeatureFrame& F : *Frames)
    {
        if (F.AudioBufferEndSec <= StartSec)
        {
            continue;
        }
        if (F.AudioBufferStartSec >= EndSec)
        {
            break;
        }

        const float OverlapStart = FMath::Max(F.AudioBufferStartSec, StartSec);
        const float OverlapEnd = FMath::Min(F.AudioBufferEndSec, EndSec);
        const float Dt = FMath::Max(OverlapEnd - OverlapStart, 0.0f);
        if (Dt <= 0.0f)
        {
            continue;
        }

        if (RuntimeSpeechFrame(F))
        {
            SpeechAccum += Dt;
            CurrentRun += Dt;
            LongestRun = FMath::Max(LongestRun, CurrentRun);
        }
        else if (RuntimeLowEvidenceFrame(F))
        {
            CurrentRun = 0.0f;
        }
        // Ambiguous frames do not reset the run; they are common inside vowels
        // and should not make sustained speech look like a pause.
    }

    return SpeechAccum >= MinSpeechSec || LongestRun >= FMath::Min(MinSpeechSec, 0.120f);
}

static bool HasLongRuntimeSilenceBetween(
    const TArray<FOffgridAIStreamingAudioFeatureFrame>* Frames,
    float StartSec,
    float EndSec,
    float MinSilenceSec)
{
    if (!Frames || EndSec <= StartSec + 0.020f)
    {
        return false;
    }

    float LowRun = 0.0f;
    for (const FOffgridAIStreamingAudioFeatureFrame& F : *Frames)
    {
        if (F.AudioBufferEndSec <= StartSec)
        {
            continue;
        }
        if (F.AudioBufferStartSec >= EndSec)
        {
            break;
        }

        const float OverlapStart = FMath::Max(F.AudioBufferStartSec, StartSec);
        const float OverlapEnd = FMath::Min(F.AudioBufferEndSec, EndSec);
        const float Dt = FMath::Max(OverlapEnd - OverlapStart, 0.0f);
        if (Dt <= 0.0f)
        {
            continue;
        }

        if (RuntimeLowEvidenceFrame(F))
        {
            LowRun += Dt;
            if (LowRun >= MinSilenceSec)
            {
                return true;
            }
        }
        else if (RuntimeSpeechFrame(F))
        {
            LowRun = 0.0f;
        }
    }
    return false;
}

static bool HasVisibleAlignedPhone(const FOffgridAIOnlinePhoneAlignmentResult& A, int32 PhoneIndex)
{
    return A.PhoneCenterSeconds.IsValidIndex(PhoneIndex) && A.PhoneCenterSeconds[PhoneIndex] >= 0.0f;
}

static bool TryGetObservedSpeechEnvelope(
    const TArray<FOffgridAIStreamingSpeechIsland>* Islands,
    float ObservedAudioEndSec,
    float& OutStartSec,
    float& OutEndSec)
{
    if (!Islands || Islands->Num() == 0)
    {
        return false;
    }

    float Start = TNumericLimits<float>::Max();
    float End = -1.0f;
    for (const FOffgridAIStreamingSpeechIsland& Island : *Islands)
    {
        const float IslandStart = Island.AudioBufferStartSec;
        float IslandEnd = FMath::Max(Island.AudioBufferLastSpeechSec, Island.AudioBufferEndSec);
        if (ObservedAudioEndSec > 0.0f)
        {
            IslandEnd = FMath::Min(IslandEnd, ObservedAudioEndSec);
        }
        if (IslandEnd <= IslandStart + 0.020f)
        {
            continue;
        }

        Start = FMath::Min(Start, IslandStart);
        End = FMath::Max(End, IslandEnd);
    }

    if (End <= Start || Start == TNumericLimits<float>::Max())
    {
        return false;
    }

    OutStartSec = Start;
    OutEndSec = End;
    return true;
}


static bool PhoneSpanOverlapsObservedSpeech(
    const TArray<FOffgridAIStreamingSpeechIsland>* Islands,
    float PhoneStartSec,
    float PhoneEndSec,
    float MarginSec)
{
    if (!Islands || Islands->Num() == 0 || PhoneEndSec <= PhoneStartSec)
    {
        return false;
    }

    for (const FOffgridAIStreamingSpeechIsland& Island : *Islands)
    {
        const float IslandStart = Island.AudioBufferStartSec - MarginSec;
        const float IslandEnd = FMath::Max(Island.AudioBufferLastSpeechSec, Island.AudioBufferEndSec) + MarginSec;
        if (IslandEnd <= IslandStart)
        {
            continue;
        }
        if (PhoneEndSec >= IslandStart && PhoneStartSec <= IslandEnd)
        {
            return true;
        }
    }
    return false;
}

struct FOffgridAIRegionWordSpan
{
    int32 WordIndex = INDEX_NONE;
    float StartSec = 0.0f;
    float EndSec = 0.0f;
};

static bool TryGetObservedSpeechRegionForSentence(
    const TArray<FOffgridAIStreamingSpeechIsland>* Islands,
    int32 SentenceCount,
    int32 SentenceIndex,
    float ObservedAudioEndSec,
    float& OutStartSec,
    float& OutEndSec)
{
    if (!Islands || SentenceIndex < 0 || SentenceCount <= 0 || Islands->Num() == 0)
    {
        return false;
    }

    int32 IslandIndex = INDEX_NONE;
    if (Islands->Num() == 1 && SentenceCount == 1)
    {
        IslandIndex = 0;
    }
    else if (Islands->Num() == SentenceCount)
    {
        IslandIndex = SentenceIndex;
    }
    else
    {
        // Advisory progress should degrade gracefully when VAD splits or merges
        // a region differently from punctuation.  Use a proportional fallback
        // instead of dropping the row entirely; the row's coverage/confidence
        // will reveal if this mapping is weak.
        const float Denom = FMath::Max(static_cast<float>(SentenceCount - 1), 1.0f);
        const float Alpha = FMath::Clamp(static_cast<float>(SentenceIndex) / Denom, 0.0f, 1.0f);
        IslandIndex = FMath::Clamp(FMath::RoundToInt(Alpha * static_cast<float>(Islands->Num() - 1)), 0, Islands->Num() - 1);
    }

    const FOffgridAIStreamingSpeechIsland& Island = (*Islands)[IslandIndex];
    const float Start = Island.AudioBufferStartSec;
    float End = FMath::Max(Island.AudioBufferLastSpeechSec, Island.AudioBufferEndSec);
    if (ObservedAudioEndSec > 0.0f)
    {
        End = FMath::Min(End, ObservedAudioEndSec);
    }
    if (End <= Start + 0.080f)
    {
        return false;
    }

    OutStartSec = Start;
    OutEndSec = End;
    return true;
}

static float WordExpectedWeight(const FOffgridAITextVisemePlan& Plan, int32 WordIndex)
{
    float Weight = 0.0f;
    for (const FOffgridAIExpectedPhone& P : Plan.ExpectedPhones)
    {
        if (P.WordIndex == WordIndex)
        {
            Weight += FMath::Max(P.WeightSeconds, 0.030f);
        }
    }
    return FMath::Max(Weight, 0.080f);
}

static float BoundaryValleyEvidence(
    const TArray<FOffgridAIStreamingAudioFeatureFrame>* Frames,
    float WindowStartSec,
    float WindowEndSec,
    float& OutValleySec,
    bool& bOutFound)
{
    bOutFound = false;
    OutValleySec = 0.5f * (WindowStartSec + WindowEndSec);
    if (!Frames || WindowEndSec <= WindowStartSec)
    {
        return 1.0f;
    }

    float BestScore = TNumericLimits<float>::Max();
    float BestTime = OutValleySec;
    for (const FOffgridAIStreamingAudioFeatureFrame& F : *Frames)
    {
        if (F.AudioBufferCenterSec < WindowStartSec)
        {
            continue;
        }
        if (F.AudioBufferCenterSec > WindowEndSec)
        {
            break;
        }

        const float Evidence = RuntimeSpeechEvidenceScore(F);
        // Use energy as a soft valley cue, but add a small preference for the
        // detector's local valley flag and for places with spectral/voicing change.
        float Score = Evidence;
        if (F.bLocalRMSValley)
        {
            Score -= 0.040f;
        }
        if (F.bLocalFluxPeak)
        {
            Score -= 0.018f;
        }
        if (Score < BestScore)
        {
            BestScore = Score;
            BestTime = F.AudioBufferCenterSec;
            bOutFound = true;
        }
    }

    OutValleySec = BestTime;
    return bOutFound ? FMath::Clamp(BestScore, 0.0f, 1.0f) : 1.0f;
}


static float AudioProgressArticulatoryTransitionProbabilityAround(
    const TArray<FOffgridAIStreamingAudioFeatureFrame>* Frames,
    float BoundarySec,
    float WindowSec,
    float* OutRedHerringProbability = nullptr)
{
    if (OutRedHerringProbability)
    {
        *OutRedHerringProbability = 1.0f;
    }
    if (!Frames || Frames->Num() == 0 || BoundarySec < 0.0f)
    {
        return 0.0f;
    }

    const float Radius = FMath::Max(WindowSec, 0.040f);
    const float WindowStart = BoundarySec - Radius;
    const float WindowEnd = BoundarySec + Radius;

    float BestTransition = 0.0f;
    float BestClosure = 0.0f;
    float BestRelease = 0.0f;
    float BestRedHerring = 1.0f;
    float SpeechSum = 0.0f;
    float VowelSum = 0.0f;
    float TransitionSum = 0.0f;
    float RedHerringSum = 0.0f;
    float BeforeVoiced = 0.0f;
    float AfterVoiced = 0.0f;
    float BeforeVowel = 0.0f;
    float AfterVowel = 0.0f;
    float BeforeFricative = 0.0f;
    float AfterFricative = 0.0f;
    float BeforeClosure = 0.0f;
    float AfterClosure = 0.0f;
    int32 Count = 0;
    int32 BeforeCount = 0;
    int32 AfterCount = 0;

    for (const FOffgridAIStreamingAudioFeatureFrame& F : *Frames)
    {
        if (F.AudioBufferCenterSec < WindowStart) continue;
        if (F.AudioBufferCenterSec > WindowEnd) break;

        const FOffgridAIArticulatoryProbabilityField A = FOffgridAIOnlinePhoneAligner::BuildArticulatoryProbabilityField(F);
        BestTransition = FMath::Max(BestTransition, A.Transition);
        BestClosure = FMath::Max(BestClosure, A.Closure);
        BestRelease = FMath::Max(BestRelease, A.Release);
        BestRedHerring = FMath::Min(BestRedHerring, A.RedHerring);
        SpeechSum += A.Speech;
        VowelSum += A.Vowel;
        TransitionSum += A.Transition;
        RedHerringSum += A.RedHerring;
        ++Count;

        if (F.AudioBufferCenterSec <= BoundarySec)
        {
            BeforeVoiced += A.Voiced;
            BeforeVowel += A.Vowel;
            BeforeFricative += A.Fricative;
            BeforeClosure += A.Closure;
            ++BeforeCount;
        }
        else
        {
            AfterVoiced += A.Voiced;
            AfterVowel += A.Vowel;
            AfterFricative += A.Fricative;
            AfterClosure += A.Closure;
            ++AfterCount;
        }
    }

    if (Count <= 0)
    {
        return 0.0f;
    }
    const float InvCount = 1.0f / static_cast<float>(Count);
    const float MeanSpeech = SpeechSum * InvCount;
    const float MeanVowel = VowelSum * InvCount;
    const float MeanTransition = TransitionSum * InvCount;
    const float MeanRedHerring = RedHerringSum * InvCount;

    if (BeforeCount > 0)
    {
        const float Inv = 1.0f / static_cast<float>(BeforeCount);
        BeforeVoiced *= Inv;
        BeforeVowel *= Inv;
        BeforeFricative *= Inv;
        BeforeClosure *= Inv;
    }
    if (AfterCount > 0)
    {
        const float Inv = 1.0f / static_cast<float>(AfterCount);
        AfterVoiced *= Inv;
        AfterVowel *= Inv;
        AfterFricative *= Inv;
        AfterClosure *= Inv;
    }

    const float StateDelta = FMath::Clamp(
        FMath::Abs(AfterVoiced - BeforeVoiced) * 0.28f +
        FMath::Abs(AfterVowel - BeforeVowel) * 0.24f +
        FMath::Abs(AfterFricative - BeforeFricative) * 0.24f +
        FMath::Abs(AfterClosure - BeforeClosure) * 0.24f,
        0.0f, 1.0f);

    const float SustainedVowelPenalty = FMath::Clamp((MeanVowel - 0.52f) / 0.38f, 0.0f, 1.0f) *
        FMath::Clamp((0.36f - StateDelta) / 0.36f, 0.0f, 1.0f);
    const float RedHerring = FMath::Clamp(0.52f * MeanRedHerring + 0.32f * SustainedVowelPenalty + 0.16f * BestRedHerring, 0.0f, 1.0f);
    if (OutRedHerringProbability)
    {
        *OutRedHerringProbability = RedHerring;
    }

    const float LexicalTransition = FMath::Clamp(
        0.32f * BestTransition +
        0.22f * MeanTransition +
        0.18f * StateDelta +
        0.14f * BestClosure +
        0.10f * BestRelease +
        0.04f * MeanSpeech,
        0.0f, 1.0f);

    return FMath::Clamp(LexicalTransition * (1.0f - 0.70f * RedHerring), 0.0f, 1.0f);
}

static float AudioProgressLexicalTransitionScoreAround(
    const TArray<FOffgridAIStreamingAudioFeatureFrame>* Frames,
    float BoundarySec,
    float WindowSec,
    float* OutRedHerringProbability = nullptr)
{
    // Compatibility wrapper: lexical-transition hints are now derived from the
    // unified articulatory probability field rather than from a separate boundary
    // detector.
    return AudioProgressArticulatoryTransitionProbabilityAround(Frames, BoundarySec, WindowSec, OutRedHerringProbability);
}


static float AudioProgressBoundaryConfidenceAround(
    const TArray<FOffgridAIStreamingAudioFeatureFrame>* Frames,
    float BoundarySec,
    float WindowSec)
{
    return AudioProgressLexicalTransitionScoreAround(Frames, BoundarySec, WindowSec, nullptr);
}

static float AudioProgressFilteredBoundaryConfidenceAround(
    const TArray<FOffgridAIStreamingAudioFeatureFrame>* Frames,
    float BoundarySec,
    float WindowSec,
    float* OutRedHerringProbability = nullptr)
{
    return AudioProgressLexicalTransitionScoreAround(Frames, BoundarySec, WindowSec, OutRedHerringProbability);
}


static float ProgressEstimatorLocalDensity(
    const TArray<FOffgridAIStreamingAudioFeatureFrame>* Frames,
    float TimeSec);



static float AudioProgressDensityProgress01(
    const TArray<FOffgridAIStreamingAudioFeatureFrame>* Frames,
    float RegionStartSec,
    float RegionEndSec,
    float CurrentSec)
{
    if (!Frames || Frames->Num() == 0 || RegionEndSec <= RegionStartSec)
    {
        return FMath::Clamp((CurrentSec - RegionStartSec) / FMath::Max(RegionEndSec - RegionStartSec, 0.001f), 0.0f, 1.0f);
    }
    float Total = 0.0f;
    float Before = 0.0f;
    float LastTime = RegionStartSec;
    for (const FOffgridAIStreamingAudioFeatureFrame& F : *Frames)
    {
        if (F.AudioBufferCenterSec < RegionStartSec) continue;
        if (F.AudioBufferCenterSec > RegionEndSec) break;
        const float Step = FMath::Clamp(F.AudioBufferCenterSec - LastTime, 0.006f, 0.030f);
        const float D = ProgressEstimatorLocalDensity(Frames, F.AudioBufferCenterSec) * Step;
        Total += D;
        if (F.AudioBufferCenterSec <= CurrentSec)
        {
            Before += D;
        }
        LastTime = F.AudioBufferCenterSec;
    }
    if (Total <= 0.0001f)
    {
        return FMath::Clamp((CurrentSec - RegionStartSec) / FMath::Max(RegionEndSec - RegionStartSec, 0.001f), 0.0f, 1.0f);
    }
    return FMath::Clamp(Before / Total, 0.0f, 1.0f);
}

static float AudioProgressPhoneExpectationProgress01(
    const FOffgridAITextVisemePlan& Plan,
    const TArray<FOffgridAIStreamingAudioFeatureFrame>* Frames,
    int32 SentenceIndex,
    float RegionStartSec,
    float RegionEndSec,
    float CurrentSec)
{
    if (RegionEndSec <= RegionStartSec)
    {
        return 0.0f;
    }
    float TotalWeight = 0.0f;
    float MatchedWeight = 0.0f;
    for (const FOffgridAIExpectedPhone& P : Plan.ExpectedPhones)
    {
        if (P.SentenceIslandIndex != SentenceIndex)
        {
            continue;
        }
        const float W = FMath::Max(P.WeightSeconds, 0.015f);
        TotalWeight += W;
        const float PhonePrior = TotalWeight / FMath::Max(TotalWeight + W, 0.001f);
        const float PhoneTime = FMath::Lerp(RegionStartSec, RegionEndSec, PhonePrior);
        if (PhoneTime <= CurrentSec)
        {
            EOffgridAIPhoneClass Class = FOffgridAIOnlinePhoneAligner::ClassForPhoneBase(P.BasePhone);
            float LocalScore = 0.5f;
            if (Frames && Frames->Num() > 0)
            {
                float BestScore = 0.0f;
                for (const FOffgridAIStreamingAudioFeatureFrame& F : *Frames)
                {
                    if (F.AudioBufferCenterSec < PhoneTime - 0.055f) continue;
                    if (F.AudioBufferCenterSec > PhoneTime + 0.055f) break;
                    const FOffgridAIPhoneClassScores Scores = FOffgridAIOnlinePhoneAligner::ScoreFramePhoneClasses(F);
                    BestScore = FMath::Max(BestScore, FOffgridAIOnlinePhoneAligner::ScoreForClass(Scores, Class));
                }
                LocalScore = FMath::Clamp(BestScore, 0.0f, 1.0f);
            }
            MatchedWeight += W * FMath::Clamp(0.35f + 0.65f * LocalScore, 0.0f, 1.0f);
        }
    }
    return FMath::Clamp(MatchedWeight / FMath::Max(TotalWeight, 0.001f), 0.0f, 1.0f);
}

struct FOffgridAIPrerollWindowLandscape
{
    float WindowStartSec = 0.0f;
    float WindowEndSec = 0.0f;
    float Coverage01 = 0.0f;
    float PriorStartProgress01 = 0.0f;
    float PriorEndProgress01 = 0.0f;
    float PosteriorStartProgress01 = 0.0f;
    float PosteriorEndProgress01 = 0.0f;
    float BestPhoneProbability = 0.0f;
    float MeanPhoneProbability = 0.0f;
    float BestBoundaryProbability = 0.0f;
    float MeanBoundaryProbability = 0.0f;
    float MeanSpeechProbability = 0.0f;
    float MatchConfidence = 0.0f;
    float SequenceMatchScore = 0.0f;
    float SequencePriorScore = 0.0f;
    float SequenceScoreGap = 0.0f;
    float SequenceStartProgress01 = 0.0f;
    float SequenceEndProgress01 = 0.0f;
    float SequenceOffsetMs = 0.0f;
    float PosteriorMeanProgress01 = 0.0f;
    float PosteriorStdDevProgress01 = 0.0f;
    float PosteriorPeakProgress01 = 0.0f;
    float PosteriorPeakProbability = 0.0f;
    float PosteriorEntropy01 = 0.0f;
    float ExpectedStreamScore = 0.0f;
    int32 PosteriorPeakPhoneIndex = INDEX_NONE;
    float CommitSafeEndSec = 0.0f;
};

static float ExpectedPhoneProgress01InRegion(
    const FOffgridAITextVisemePlan& Plan,
    int32 SentenceIndex,
    int32 PhoneIndex)
{
    float Total = 0.0f;
    float Before = 0.0f;
    for (const FOffgridAIExpectedPhone& P : Plan.ExpectedPhones)
    {
        if (P.SentenceIslandIndex != SentenceIndex)
        {
            continue;
        }
        const float W = FMath::Max(P.WeightSeconds, 0.015f);
        if (P.PhoneIndex < PhoneIndex)
        {
            Before += W;
        }
        Total += W;
    }
    return FMath::Clamp((Before + 0.5f * 0.075f) / FMath::Max(Total, 0.001f), 0.0f, 1.0f);
}

static const FOffgridAIExpectedPhone* FindExpectedPhoneNearRegionProgress(
    const FOffgridAITextVisemePlan& Plan,
    int32 SentenceIndex,
    float RegionProgress01)
{
    const FOffgridAIExpectedPhone* Best = nullptr;
    float BestDistance = TNumericLimits<float>::Max();
    float Total = 0.0f;
    for (const FOffgridAIExpectedPhone& P : Plan.ExpectedPhones)
    {
        if (P.SentenceIslandIndex == SentenceIndex)
        {
            Total += FMath::Max(P.WeightSeconds, 0.015f);
        }
    }
    if (Total <= 0.001f)
    {
        return nullptr;
    }
    float Accum = 0.0f;
    for (const FOffgridAIExpectedPhone& P : Plan.ExpectedPhones)
    {
        if (P.SentenceIslandIndex != SentenceIndex)
        {
            continue;
        }
        const float W = FMath::Max(P.WeightSeconds, 0.015f);
        const float Center = (Accum + 0.5f * W) / Total;
        const float D = FMath::Abs(Center - RegionProgress01);
        if (D < BestDistance)
        {
            BestDistance = D;
            Best = &P;
        }
        Accum += W;
    }
    return Best;
}



struct FOffgridAIExpectedArticulationSample
{
    float Speech = 1.0f;
    float Vowel = 0.0f;
    float Closure = 0.0f;
    float Release = 0.0f;
    float Fricative = 0.0f;
    float Voiced = 0.0f;
    float LowTilt = 0.0f;
    float Transition = 0.0f;
    EOffgridAIPhoneClass DominantClass = EOffgridAIPhoneClass::Unknown;
};

static void ExpectedArticulationForPhoneClass(EOffgridAIPhoneClass Class, FOffgridAIExpectedArticulationSample& Out, float Weight)
{
    Out.Speech += Weight;
    switch (Class)
    {
    case EOffgridAIPhoneClass::Bilabial:
        Out.Closure += 0.86f * Weight;
        Out.Release += 0.55f * Weight;
        Out.Transition += 0.55f * Weight;
        break;
    case EOffgridAIPhoneClass::Labiodental:
        Out.Fricative += 0.82f * Weight;
        Out.Transition += 0.38f * Weight;
        break;
    case EOffgridAIPhoneClass::Sibilant:
        Out.Fricative += 0.92f * Weight;
        Out.Transition += 0.34f * Weight;
        break;
    case EOffgridAIPhoneClass::Glide:
        Out.Vowel += 0.46f * Weight;
        Out.Voiced += 0.74f * Weight;
        Out.LowTilt += 0.52f * Weight;
        break;
    case EOffgridAIPhoneClass::VowelOpen:
        Out.Vowel += 0.92f * Weight;
        Out.Voiced += 0.82f * Weight;
        break;
    case EOffgridAIPhoneClass::VowelFront:
        Out.Vowel += 0.88f * Weight;
        Out.Voiced += 0.84f * Weight;
        Out.LowTilt += 0.18f * Weight;
        break;
    case EOffgridAIPhoneClass::VowelRound:
        Out.Vowel += 0.88f * Weight;
        Out.Voiced += 0.82f * Weight;
        Out.LowTilt += 0.72f * Weight;
        break;
    default:
        Out.Voiced += 0.30f * Weight;
        break;
    }
}

static FOffgridAIExpectedArticulationSample ExpectedArticulationAtRegionProgress(
    const FOffgridAITextVisemePlan& Plan,
    int32 SentenceIndex,
    float RegionProgress01)
{
    FOffgridAIExpectedArticulationSample Out;
    float Total = 0.0f;
    for (const FOffgridAIExpectedPhone& P : Plan.ExpectedPhones)
    {
        if (P.SentenceIslandIndex == SentenceIndex)
        {
            Total += FMath::Max(P.WeightSeconds, 0.015f);
        }
    }
    if (Total <= 0.001f)
    {
        return Out;
    }

    float Accum = 0.0f;
    float WeightSum = 0.0f;
    float BestWeight = -1.0f;
    for (const FOffgridAIExpectedPhone& P : Plan.ExpectedPhones)
    {
        if (P.SentenceIslandIndex != SentenceIndex)
        {
            continue;
        }
        const float WSec = FMath::Max(P.WeightSeconds, 0.015f);
        const float Center = (Accum + 0.5f * WSec) / Total;
        const float Width = FMath::Clamp(0.5f * WSec / Total + 0.018f, 0.020f, 0.090f);
        const float D = (RegionProgress01 - Center) / Width;
        const float W = std::exp(-0.5f * D * D);
        const EOffgridAIPhoneClass Class = FOffgridAIOnlinePhoneAligner::ClassForPhoneBase(P.BasePhone);
        ExpectedArticulationForPhoneClass(Class, Out, W);
        if (W > BestWeight)
        {
            BestWeight = W;
            Out.DominantClass = Class;
        }
        WeightSum += W;
        Accum += WSec;
    }

    if (WeightSum > 0.0001f)
    {
        Out.Speech = FMath::Clamp(Out.Speech / WeightSum, 0.0f, 1.0f);
        Out.Vowel = FMath::Clamp(Out.Vowel / WeightSum, 0.0f, 1.0f);
        Out.Closure = FMath::Clamp(Out.Closure / WeightSum, 0.0f, 1.0f);
        Out.Release = FMath::Clamp(Out.Release / WeightSum, 0.0f, 1.0f);
        Out.Fricative = FMath::Clamp(Out.Fricative / WeightSum, 0.0f, 1.0f);
        Out.Voiced = FMath::Clamp(Out.Voiced / WeightSum, 0.0f, 1.0f);
        Out.LowTilt = FMath::Clamp(Out.LowTilt / WeightSum, 0.0f, 1.0f);
        Out.Transition = FMath::Clamp(Out.Transition / WeightSum, 0.0f, 1.0f);
    }

    // Add a normalized boundary/transition prior at expected phone-change points.
    // This is deliberately derived from the transcript stream, not from word IDs.
    Accum = 0.0f;
    for (const FOffgridAIExpectedPhone& P : Plan.ExpectedPhones)
    {
        if (P.SentenceIslandIndex != SentenceIndex)
        {
            continue;
        }
        const float WSec = FMath::Max(P.WeightSeconds, 0.015f);
        const float Boundary = Accum / Total;
        if (Boundary > 0.001f && Boundary < 0.999f)
        {
            const float D = FMath::Abs(RegionProgress01 - Boundary) / 0.020f;
            Out.Transition = FMath::Max(Out.Transition, 0.75f * std::exp(-0.5f * D * D));
        }
        Accum += WSec;
    }
    return Out;
}

static float ScoreObservedFieldAgainstExpectedArticulation(
    const FOffgridAIArticulatoryProbabilityField& Field,
    const FOffgridAIExpectedArticulationSample& Expected)
{
    const float PhoneScore = FMath::Clamp(FOffgridAIOnlinePhoneAligner::ScoreForClass(Field.PhoneScores, Expected.DominantClass), 0.0f, 1.0f);
    const float VowelScore = 1.0f - FMath::Abs(Field.Vowel - Expected.Vowel);
    const float ClosureScore = 1.0f - FMath::Abs(Field.Closure - Expected.Closure);
    const float ReleaseScore = 1.0f - FMath::Abs(Field.Release - Expected.Release);
    const float FricativeScore = 1.0f - FMath::Abs(Field.Fricative - Expected.Fricative);
    const float VoicedScore = 1.0f - FMath::Abs(Field.Voiced - Expected.Voiced);
    const float RoundTiltScore = 1.0f - FMath::Abs(Field.LowTilt - Expected.LowTilt);
    const float TransitionScore = 1.0f - FMath::Abs(Field.Transition - Expected.Transition);
    const float Shape = FMath::Clamp(
        0.20f * VowelScore +
        0.15f * ClosureScore +
        0.10f * ReleaseScore +
        0.14f * FricativeScore +
        0.14f * VoicedScore +
        0.10f * RoundTiltScore +
        0.09f * TransitionScore +
        0.08f * Field.Speech,
        0.0f, 1.0f);
    return FMath::Clamp(0.54f * Shape + 0.46f * PhoneScore - 0.18f * Field.RedHerring, 0.0f, 1.0f);
}
static float ScoreObservedFieldAgainstExpectedPhone(
    const FOffgridAIArticulatoryProbabilityField& Field,
    const FOffgridAIExpectedPhone* Expected)
{
    if (!Expected)
    {
        return 0.0f;
    }

    const EOffgridAIPhoneClass ExpectedClass = FOffgridAIOnlinePhoneAligner::ClassForPhoneBase(Expected->BasePhone);
    float PhoneScore = FMath::Clamp(FOffgridAIOnlinePhoneAligner::ScoreForClass(Field.PhoneScores, ExpectedClass), 0.0f, 1.0f);

    // Blend class likelihood with broad articulatory expectations so the
    // sequence matcher rewards the shape of the trajectory, not just a single
    // hard phone label.  The CMU plan supplies the prior trajectory; the rolling
    // probability field supplies the observed trajectory.
    const bool bExpectedVowel = (ExpectedClass == EOffgridAIPhoneClass::VowelOpen || ExpectedClass == EOffgridAIPhoneClass::VowelFront || ExpectedClass == EOffgridAIPhoneClass::VowelRound);
    const bool bExpectedClosure = (ExpectedClass == EOffgridAIPhoneClass::Bilabial);
    const bool bExpectedFricative = (ExpectedClass == EOffgridAIPhoneClass::Labiodental || ExpectedClass == EOffgridAIPhoneClass::Sibilant);
    const bool bExpectedRound = (ExpectedClass == EOffgridAIPhoneClass::VowelRound || ExpectedClass == EOffgridAIPhoneClass::Glide);

    float ShapeScore = PhoneScore;
    if (bExpectedVowel)
    {
        ShapeScore = FMath::Max(ShapeScore, 0.55f * Field.Vowel + 0.25f * Field.Voiced + 0.20f * Field.Speech);
    }
    if (bExpectedClosure)
    {
        ShapeScore = FMath::Max(ShapeScore, 0.55f * Field.Closure + 0.25f * Field.Release + 0.20f * (1.0f - Field.Vowel));
    }
    if (bExpectedFricative)
    {
        ShapeScore = FMath::Max(ShapeScore, 0.60f * Field.Fricative + 0.25f * Field.Transition + 0.15f * Field.Speech);
    }
    if (bExpectedRound)
    {
        // The public field stores broad spectral tilt rather than a dedicated rounding value.
        // LowTilt is a reasonable proxy for rounded/back-vowel energy in this lightweight matcher.
        ShapeScore = FMath::Max(ShapeScore, 0.45f * Field.LowTilt + 0.35f * Field.Vowel + 0.20f * Field.Speech);
    }

    const float RedHerringPenalty = 0.20f * Field.RedHerring;
    return FMath::Clamp(0.68f * PhoneScore + 0.32f * ShapeScore - RedHerringPenalty, 0.0f, 1.0f);
}


struct FOffgridAIPrerollScoringSample
{
    float OffsetSec = 0.0f;
    FOffgridAIArticulatoryProbabilityField Field;
};

static void BuildPrerollScoringSamples(
    const TArray<FOffgridAIStreamingAudioFeatureFrame>* Frames,
    float WindowStartSec,
    float WindowEndSec,
    TArray<FOffgridAIPrerollScoringSample>& OutSamples)
{
    OutSamples.Reset();
    if (!Frames || Frames->Num() == 0 || WindowEndSec <= WindowStartSec)
    {
        return;
    }

    int32 FirstFrame = INDEX_NONE;
    int32 LastFrame = INDEX_NONE;
    for (int32 I = 0; I < Frames->Num(); ++I)
    {
        const float T = (*Frames)[I].AudioBufferCenterSec;
        if (T < WindowStartSec)
        {
            continue;
        }
        if (T > WindowEndSec)
        {
            break;
        }
        if (FirstFrame == INDEX_NONE)
        {
            FirstFrame = I;
        }
        LastFrame = I;
    }

    if (FirstFrame == INDEX_NONE || LastFrame < FirstFrame)
    {
        return;
    }

    const float WindowDuration = FMath::Max(WindowEndSec - WindowStartSec, 0.001f);
    const int32 AvailableFrames = LastFrame - FirstFrame + 1;
    const int32 MaxScoringFrames = WindowDuration > 1.500f ? 120 : 96;
    const int32 FrameStride = FMath::Max(1, (AvailableFrames + MaxScoringFrames - 1) / MaxScoringFrames);

    for (int32 I = FirstFrame; I <= LastFrame; I += FrameStride)
    {
        const FOffgridAIStreamingAudioFeatureFrame& F = (*Frames)[I];
        FOffgridAIPrerollScoringSample Sample;
        Sample.OffsetSec = F.AudioBufferCenterSec - WindowStartSec;
        Sample.Field = FOffgridAIOnlinePhoneAligner::BuildArticulatoryProbabilityField(F);
        OutSamples.Add(Sample);
    }
}

static float ScorePrerollSamplesAtRegionProgress(
    const FOffgridAITextVisemePlan& Plan,
    const TArray<FOffgridAIPrerollScoringSample>& Samples,
    int32 SentenceIndex,
    float RegionStartSec,
    float RegionEndSec,
    float CandidateStartProgress01,
    float* OutCoverage)
{
    if (OutCoverage)
    {
        *OutCoverage = 0.0f;
    }
    if (Samples.Num() == 0 || RegionEndSec <= RegionStartSec)
    {
        return 0.0f;
    }

    const float RegionDuration = FMath::Max(RegionEndSec - RegionStartSec, 0.001f);
    float WeightedScore = 0.0f;
    float WeightSum = 0.0f;
    float CoveredFrames = 0.0f;

    for (const FOffgridAIPrerollScoringSample& Sample : Samples)
    {
        const float CandidateProgress = CandidateStartProgress01 + Sample.OffsetSec / RegionDuration;
        if (CandidateProgress < 0.0f || CandidateProgress > 1.0f)
        {
            continue;
        }

        const FOffgridAIExpectedArticulationSample Expected = ExpectedArticulationAtRegionProgress(Plan, SentenceIndex, CandidateProgress);
        const float SpeechWeight = FMath::Clamp(0.18f + 0.82f * FMath::Max(Sample.Field.Speech, Expected.Speech), 0.08f, 1.0f);
        const float Score = ScoreObservedFieldAgainstExpectedArticulation(Sample.Field, Expected);
        WeightedScore += Score * SpeechWeight;
        WeightSum += SpeechWeight;
        CoveredFrames += 1.0f;
    }

    if (OutCoverage)
    {
        *OutCoverage = CoveredFrames / static_cast<float>(FMath::Max(Samples.Num(), 1));
    }
    return WeightSum > 0.001f ? FMath::Clamp(WeightedScore / WeightSum, 0.0f, 1.0f) : 0.0f;
}

static float ScorePrerollWindowAtRegionProgress(
    const FOffgridAITextVisemePlan& Plan,
    const TArray<FOffgridAIStreamingAudioFeatureFrame>* Frames,
    int32 SentenceIndex,
    float RegionStartSec,
    float RegionEndSec,
    float WindowStartSec,
    float WindowEndSec,
    float CandidateStartProgress01,
    float* OutCoverage)
{
    if (OutCoverage)
    {
        *OutCoverage = 0.0f;
    }
    if (!Frames || Frames->Num() == 0 || WindowEndSec <= WindowStartSec || RegionEndSec <= RegionStartSec)
    {
        return 0.0f;
    }

    // Compare the observed rolling preroll window to a normalized expected
    // articulation stream for the whole current speech region.  Long preroll
    // windows are deliberately sampled coarsely: extra future context should
    // improve stability, not make every 10 ms tick linearly more expensive.
    const float RegionDuration = FMath::Max(RegionEndSec - RegionStartSec, 0.001f);
    const float WindowDuration = FMath::Max(WindowEndSec - WindowStartSec, 0.001f);

    int32 FirstFrame = INDEX_NONE;
    int32 LastFrame = INDEX_NONE;
    for (int32 I = 0; I < Frames->Num(); ++I)
    {
        const float T = (*Frames)[I].AudioBufferCenterSec;
        if (T < WindowStartSec)
        {
            continue;
        }
        if (T > WindowEndSec)
        {
            break;
        }
        if (FirstFrame == INDEX_NONE)
        {
            FirstFrame = I;
        }
        LastFrame = I;
    }

    if (FirstFrame == INDEX_NONE || LastFrame < FirstFrame)
    {
        return 0.0f;
    }

    const int32 AvailableFrames = LastFrame - FirstFrame + 1;
    // Dense enough for short/normal preroll, bounded for large offline sweeps.
    // A 10s preroll should use roughly 120 samples, not 1000 samples, per
    // candidate cursor.
    const int32 MaxScoringFrames = WindowDuration > 1.500f ? 120 : 96;
    const int32 FrameStride = FMath::Max(1, (AvailableFrames + MaxScoringFrames - 1) / MaxScoringFrames);

    float WeightedScore = 0.0f;
    float WeightSum = 0.0f;
    float CoveredFrames = 0.0f;
    float TotalSampledFrames = 0.0f;

    for (int32 I = FirstFrame; I <= LastFrame; I += FrameStride)
    {
        const FOffgridAIStreamingAudioFeatureFrame& F = (*Frames)[I];
        TotalSampledFrames += 1.0f;

        const float OffsetSec = F.AudioBufferCenterSec - WindowStartSec;
        const float CandidateProgress = CandidateStartProgress01 + OffsetSec / RegionDuration;
        if (CandidateProgress < 0.0f || CandidateProgress > 1.0f)
        {
            continue;
        }

        const FOffgridAIArticulatoryProbabilityField Field = FOffgridAIOnlinePhoneAligner::BuildArticulatoryProbabilityField(F);
        const FOffgridAIExpectedArticulationSample Expected = ExpectedArticulationAtRegionProgress(Plan, SentenceIndex, CandidateProgress);
        const float SpeechWeight = FMath::Clamp(0.18f + 0.82f * FMath::Max(Field.Speech, Expected.Speech), 0.08f, 1.0f);
        const float Score = ScoreObservedFieldAgainstExpectedArticulation(Field, Expected);
        WeightedScore += Score * SpeechWeight;
        WeightSum += SpeechWeight;
        CoveredFrames += 1.0f;
    }

    if (OutCoverage && TotalSampledFrames > 0.0f)
    {
        *OutCoverage = CoveredFrames / TotalSampledFrames;
    }
    return WeightSum > 0.001f ? FMath::Clamp(WeightedScore / WeightSum, 0.0f, 1.0f) : 0.0f;
}

static bool AddUniqueCursorIndex(TArray<int32>& Values, int32 Value)
{
    for (int32 Existing : Values)
    {
        if (Existing == Value)
        {
            return false;
        }
    }
    Values.Add(Value);
    return true;
}

static float CursorTrackerEvidenceCoverage01(float WindowDurationSec, float PrerollSec, float RegionDurationSec)
{
    // Coverage used for confidence should mean "how much useful speech-region
    // evidence is available," not "how much of the requested preroll budget was
    // filled."  With a 10s diagnostic preroll, most utterances are shorter than
    // the requested buffer, so WindowDuration / PrerollSec falsely reports low
    // confidence even when the tracker can see the whole remaining region.
    const float UsefulDenom = FMath::Max(FMath::Min(FMath::Max(PrerollSec, 0.001f), FMath::Max(RegionDurationSec, 0.001f)), 0.001f);
    return FMath::Clamp(WindowDurationSec / UsefulDenom, 0.0f, 1.0f);
}

static void EvaluatePrerollWindowSequenceMatch(
    const FOffgridAITextVisemePlan& Plan,
    const TArray<FOffgridAIStreamingAudioFeatureFrame>* Frames,
    int32 SentenceIndex,
    float RegionStartSec,
    float RegionEndSec,
    float WindowStartSec,
    float WindowEndSec,
    bool bHasPreviousCursor,
    int32 PreviousCursorSentenceIndex,
    float PreviousCursorProgress01,
    float PreviousCursorPlaybackSec,
    FOffgridAIPrerollWindowLandscape& Out)
{
    const float RegionDuration = FMath::Max(RegionEndSec - RegionStartSec, 0.001f);
    const float WindowDuration = FMath::Max(WindowEndSec - WindowStartSec, 0.0f);
    if (!Frames || Frames->Num() == 0 || WindowDuration <= 0.001f)
    {
        return;
    }

    // Build the observed preroll samples once per tick/window.  Earlier versions
    // rescanned the full frame list and rebuilt probability fields for every
    // candidate cursor, which made long-preroll diagnostics scale poorly.
    TArray<FOffgridAIPrerollScoringSample> ScoringSamples;
    BuildPrerollScoringSamples(Frames, WindowStartSec, WindowEndSec, ScoringSamples);
    if (ScoringSamples.Num() == 0)
    {
        return;
    }

    const float TimeCursorProgress01 = FMath::Clamp((WindowStartSec - RegionStartSec) / RegionDuration, 0.0f, 1.0f);
    const float WindowSpan01 = FMath::Clamp(WindowDuration / RegionDuration, 0.0f, 1.0f);
    const float MinCursor01 = 0.0f;
    const float MaxCursor01 = FMath::Max(0.0f, 1.0f - WindowSpan01);

    float PredictedCursor01 = TimeCursorProgress01;
    if (bHasPreviousCursor && PreviousCursorSentenceIndex == SentenceIndex)
    {
        const float Dt = FMath::Clamp(WindowStartSec - PreviousCursorPlaybackSec, -0.250f, 0.250f);
        PredictedCursor01 = FMath::Clamp(PreviousCursorProgress01 + Dt / RegionDuration, MinCursor01, MaxCursor01);
    }
    // The playback clock remains a weak global sanity prior, but the hidden
    // state is now a cursor over the expected articulation stream.  This is the
    // single inference object replacing the earlier progress/density/boundary/
    // phone estimators.
    const float PriorCursor01 = FMath::Clamp(0.18f * TimeCursorProgress01 + 0.82f * PredictedCursor01, MinCursor01, MaxCursor01);

    // Build a discrete cursor lattice over the precomputed normalized stream.
    // Each cursor means: "the current preroll window starts at this point in the
    // CMU articulation stream."  The tracker evaluates a local band around the
    // predicted cursor and converts those scores into a likelihood curve.
    const int32 RawCursorCount = static_cast<int32>(RegionDuration / 0.010f) + 1;
    const int32 CursorCount = FMath::Clamp(RawCursorCount, 48, 360);
    const float CursorStep01 = CursorCount > 1 ? 1.0f / static_cast<float>(CursorCount - 1) : 1.0f;

    const int32 PriorCursorIndex = FMath::Clamp(static_cast<int32>(PriorCursor01 / CursorStep01 + 0.5f), 0, CursorCount - 1);
    const int32 TimeCursorIndex = FMath::Clamp(static_cast<int32>(TimeCursorProgress01 / CursorStep01 + 0.5f), 0, CursorCount - 1);
    const int32 MaxValidCursorIndex = FMath::Clamp(static_cast<int32>(MaxCursor01 / CursorStep01 + 0.5f), 0, CursorCount - 1);

    const int32 SearchRadiusCursors = FMath::Clamp(static_cast<int32>((FMath::Max(WindowSpan01, 0.04f) + 0.14f) / CursorStep01 + 0.5f), 8, 90);
    const int32 SearchStartCursor = FMath::Clamp(PriorCursorIndex - SearchRadiusCursors, 0, MaxValidCursorIndex);
    const int32 SearchEndCursor = FMath::Clamp(PriorCursorIndex + SearchRadiusCursors, 0, MaxValidCursorIndex);
    const int32 FullCandidateCount = FMath::Max(SearchEndCursor - SearchStartCursor + 1, 1);

    float PriorCoverage = 0.0f;
    Out.SequencePriorScore = ScorePrerollSamplesAtRegionProgress(Plan, ScoringSamples, SentenceIndex, RegionStartSec, RegionEndSec, PriorCursor01, &PriorCoverage);
    Out.ExpectedStreamScore = Out.SequencePriorScore;

    // Two-stage cursor search: coarse scan across the local band, then dense
    // refinement around the best coarse cursor plus the time/prior anchors.  This
    // keeps large-preroll diagnostics tractable while preserving local precision.
    TArray<int32> CoarseCursorIndices;
    // Keep normal streaming windows fully dense.  The coarse search exists only
    // to make very large offline preroll sweeps tractable; using it at 350 ms
    // can change the posterior for no useful performance gain.
    const bool bDenseStreamingWindow = WindowDuration <= 0.750f || FullCandidateCount <= 120;
    const int32 MaxCoarseCandidates = bDenseStreamingWindow ? FullCandidateCount : 48;
    const int32 CoarseStride = bDenseStreamingWindow ? 1 : FMath::Max(1, (FullCandidateCount + MaxCoarseCandidates - 1) / MaxCoarseCandidates);
    AddUniqueCursorIndex(CoarseCursorIndices, PriorCursorIndex);
    AddUniqueCursorIndex(CoarseCursorIndices, TimeCursorIndex);
    AddUniqueCursorIndex(CoarseCursorIndices, SearchStartCursor);
    AddUniqueCursorIndex(CoarseCursorIndices, SearchEndCursor);
    for (int32 CursorIndex = SearchStartCursor; CursorIndex <= SearchEndCursor; CursorIndex += CoarseStride)
    {
        AddUniqueCursorIndex(CoarseCursorIndices, CursorIndex);
    }

    const float PriorSigmaCursors = FMath::Max(static_cast<float>(SearchRadiusCursors) * 0.36f, 3.0f);
    const float TimeSigmaCursors = FMath::Max(static_cast<float>(SearchRadiusCursors) * 0.90f, 8.0f);
    const float ScoreTemperature = 0.052f;

    auto ScoreCursorIndex = [&](int32 CursorIndex, float& OutAcousticScore, float& OutCandidateStart) -> float
    {
        OutCandidateStart = FMath::Clamp(static_cast<float>(CursorIndex) * CursorStep01, MinCursor01, MaxCursor01);
        float CandidateCoverage = 0.0f;
        OutAcousticScore = ScorePrerollSamplesAtRegionProgress(Plan, ScoringSamples, SentenceIndex, RegionStartSec, RegionEndSec, OutCandidateStart, &CandidateCoverage);

        const float PriorDistance = static_cast<float>(CursorIndex - PriorCursorIndex);
        const float TimeDistance = static_cast<float>(CursorIndex - TimeCursorIndex);
        const float TrackerLogPrior = -0.5f * (PriorDistance / PriorSigmaCursors) * (PriorDistance / PriorSigmaCursors);
        const float TimeLogPrior = -0.5f * (TimeDistance / TimeSigmaCursors) * (TimeDistance / TimeSigmaCursors);
        const float CoveragePenalty = std::log(FMath::Clamp(0.15f + 0.85f * CandidateCoverage, 0.001f, 1.0f));
        const float AcousticLogLikelihood = (OutAcousticScore - 0.50f) / FMath::Max(ScoreTemperature, 0.001f);
        return AcousticLogLikelihood + 0.72f * TrackerLogPrior + 0.16f * TimeLogPrior + 0.55f * CoveragePenalty;
    };

    float CoarseBestLogLikelihood = -1000000.0f;
    int32 CoarseBestCursorIndex = PriorCursorIndex;
    for (int32 CursorIndex : CoarseCursorIndices)
    {
        CursorIndex = FMath::Clamp(CursorIndex, SearchStartCursor, SearchEndCursor);
        float AcousticScore = 0.0f;
        float CandidateStart = 0.0f;
        const float L = ScoreCursorIndex(CursorIndex, AcousticScore, CandidateStart);
        if (L > CoarseBestLogLikelihood)
        {
            CoarseBestLogLikelihood = L;
            CoarseBestCursorIndex = CursorIndex;
        }
    }

    TArray<int32> EvalCursorIndices;
    if (bDenseStreamingWindow)
    {
        for (int32 CursorIndex = SearchStartCursor; CursorIndex <= SearchEndCursor; ++CursorIndex)
        {
            AddUniqueCursorIndex(EvalCursorIndices, CursorIndex);
        }
    }
    else
    {
        const int32 FineRadius = FMath::Clamp(CoarseStride * 3, 4, 18);
        for (int32 CursorIndex : CoarseCursorIndices)
        {
            AddUniqueCursorIndex(EvalCursorIndices, FMath::Clamp(CursorIndex, SearchStartCursor, SearchEndCursor));
        }
        for (int32 CursorIndex = CoarseBestCursorIndex - FineRadius; CursorIndex <= CoarseBestCursorIndex + FineRadius; ++CursorIndex)
        {
            AddUniqueCursorIndex(EvalCursorIndices, FMath::Clamp(CursorIndex, SearchStartCursor, SearchEndCursor));
        }
        for (int32 CursorIndex = PriorCursorIndex - 4; CursorIndex <= PriorCursorIndex + 4; ++CursorIndex)
        {
            AddUniqueCursorIndex(EvalCursorIndices, FMath::Clamp(CursorIndex, SearchStartCursor, SearchEndCursor));
        }
    }

    const int32 CandidateCount = FMath::Max(EvalCursorIndices.Num(), 1);
    TArray<float> CandidateProgress;
    TArray<float> CandidateAcousticScore;
    TArray<float> CandidateLogLikelihood;
    CandidateProgress.SetNum(CandidateCount);
    CandidateAcousticScore.SetNum(CandidateCount);
    CandidateLogLikelihood.SetNum(CandidateCount);

    float BestLogLikelihood = -1000000.0f;
    float SecondBestLogLikelihood = -1000000.0f;
    float BestAcousticScore = 0.0f;
    float BestProgress01 = PriorCursor01;
    int32 BestLocalIndex = 0;

    for (int32 Local = 0; Local < CandidateCount; ++Local)
    {
        const int32 CursorIndex = EvalCursorIndices[Local];
        float AcousticScore = 0.0f;
        float CandidateStart = 0.0f;
        const float CombinedLogLikelihood = ScoreCursorIndex(CursorIndex, AcousticScore, CandidateStart);

        CandidateProgress[Local] = CandidateStart;
        CandidateAcousticScore[Local] = AcousticScore;
        CandidateLogLikelihood[Local] = CombinedLogLikelihood;

        if (CombinedLogLikelihood > BestLogLikelihood)
        {
            SecondBestLogLikelihood = BestLogLikelihood;
            BestLogLikelihood = CombinedLogLikelihood;
            BestAcousticScore = AcousticScore;
            BestProgress01 = CandidateStart;
            BestLocalIndex = Local;
        }
        else if (CombinedLogLikelihood > SecondBestLogLikelihood)
        {
            SecondBestLogLikelihood = CombinedLogLikelihood;
        }
    }

    float SumProb = 0.0f;
    float Mean = 0.0f;
    float Entropy = 0.0f;
    float PeakProbability = 0.0f;
    TArray<float> Probabilities;
    Probabilities.SetNum(CandidateCount);
    for (int32 I = 0; I < CandidateCount; ++I)
    {
        const float Unnorm = std::exp(FMath::Clamp(CandidateLogLikelihood[I] - BestLogLikelihood, -60.0f, 0.0f));
        Probabilities[I] = Unnorm;
        SumProb += Unnorm;
    }
    SumProb = FMath::Max(SumProb, 0.000001f);
    for (int32 I = 0; I < CandidateCount; ++I)
    {
        const float P = Probabilities[I] / SumProb;
        Probabilities[I] = P;
        Mean += P * CandidateProgress[I];
        PeakProbability = FMath::Max(PeakProbability, P);
        if (P > 0.000001f)
        {
            Entropy -= P * std::log(P);
        }
    }

    float Variance = 0.0f;
    for (int32 I = 0; I < CandidateCount; ++I)
    {
        const float D = CandidateProgress[I] - Mean;
        Variance += Probabilities[I] * D * D;
    }
    const float StdDev = std::sqrt(FMath::Max(Variance, 0.0f));
    const float MaxEntropy = std::log(static_cast<float>(FMath::Max(CandidateCount, 2)));
    const float Entropy01 = MaxEntropy > 0.0001f ? FMath::Clamp(Entropy / MaxEntropy, 0.0f, 1.0f) : 1.0f;

    Out.SequenceMatchScore = FMath::Clamp(BestAcousticScore, 0.0f, 1.0f);
    Out.SequenceScoreGap = FMath::Clamp((BestLogLikelihood - SecondBestLogLikelihood) / 4.0f, 0.0f, 1.0f);
    Out.SequenceStartProgress01 = FMath::Clamp(BestProgress01, 0.0f, 1.0f);
    Out.SequenceEndProgress01 = FMath::Clamp(BestProgress01 + WindowSpan01, 0.0f, 1.0f);
    Out.SequenceOffsetMs = (BestProgress01 - PriorCursor01) * RegionDuration * 1000.0f;

    const float RawMean = FMath::Clamp(Mean, 0.0f, 1.0f);
    const float RawPeak = FMath::Clamp(BestProgress01, 0.0f, 1.0f);
    const float Sharpness = FMath::Clamp(1.0f - Entropy01, 0.0f, 1.0f);
    const float EvidenceCoverage01 = FMath::Clamp(Out.Coverage01, 0.0f, 1.0f);
    const float PeakQuality = FMath::Clamp((PeakProbability - 0.08f) / 0.34f, 0.0f, 1.0f);
    const float GapQuality = FMath::Clamp(Out.SequenceScoreGap * 1.35f, 0.0f, 1.0f);
    const float MatchQuality = FMath::Clamp((Out.SequenceMatchScore - Out.SequencePriorScore + 0.030f) / 0.115f, 0.0f, 1.0f);

    // The cursor likelihood is useful evidence, but at 350 ms it is often broad
    // and noisy.  Treat it as a correction to the tracked/time prior, not as a
    // replacement, unless the posterior is both sharp and better than the prior.
    const float CursorCorrectionTrust = FMath::Clamp(
        0.10f + 0.38f * PeakQuality + 0.24f * Sharpness + 0.18f * GapQuality + 0.10f * MatchQuality,
        0.0f,
        0.82f) * EvidenceCoverage01;

    const float CorrectedMean = FMath::Clamp(PriorCursor01 + CursorCorrectionTrust * (RawMean - PriorCursor01), MinCursor01, MaxCursor01);

    Out.PosteriorMeanProgress01 = CorrectedMean;
    Out.PosteriorStdDevProgress01 = FMath::Clamp(StdDev, 0.0f, 1.0f);
    Out.PosteriorPeakProgress01 = RawPeak;
    Out.PosteriorPeakProbability = FMath::Clamp(PeakProbability, 0.0f, 1.0f);
    Out.PosteriorEntropy01 = Entropy01;

    const FOffgridAIExpectedPhone* PeakPhone = FindExpectedPhoneNearRegionProgress(Plan, SentenceIndex, Out.PosteriorPeakProgress01 + 0.5f * WindowSpan01);
    Out.PosteriorPeakPhoneIndex = PeakPhone ? PeakPhone->PhoneIndex : INDEX_NONE;

    Out.PosteriorStartProgress01 = Out.PosteriorMeanProgress01;
    Out.PosteriorEndProgress01 = FMath::Clamp(Out.PosteriorMeanProgress01 + WindowSpan01, 0.0f, 1.0f);

    Out.MatchConfidence = FMath::Clamp(0.25f * Out.MatchConfidence + 0.75f * CursorCorrectionTrust, 0.0f, 1.0f);
}

static FOffgridAIPrerollWindowLandscape BuildPrerollWindowLandscape(
    const FOffgridAITextVisemePlan& Plan,
    const TArray<FOffgridAIStreamingAudioFeatureFrame>* Frames,
    int32 SentenceIndex,
    float RegionStartSec,
    float RegionEndSec,
    float PlaybackSec,
    float PrerollSec,
    float ObservedAudioEndSec,
    bool bHasPreviousCursor,
    int32 PreviousCursorSentenceIndex,
    float PreviousCursorProgress01,
    float PreviousCursorPlaybackSec)
{
    FOffgridAIPrerollWindowLandscape Out;
    Out.WindowStartSec = FMath::Max(PlaybackSec, RegionStartSec);
    Out.WindowEndSec = FMath::Min(FMath::Min(PlaybackSec + FMath::Max(PrerollSec, 0.0f), ObservedAudioEndSec), RegionEndSec);
    Out.CommitSafeEndSec = FMath::Max(Out.WindowStartSec, Out.WindowEndSec - StreamingCommitLagSec);

    const float RegionDuration = FMath::Max(RegionEndSec - RegionStartSec, 0.001f);
    Out.PriorStartProgress01 = FMath::Clamp((Out.WindowStartSec - RegionStartSec) / RegionDuration, 0.0f, 1.0f);
    Out.PriorEndProgress01 = FMath::Clamp((Out.WindowEndSec - RegionStartSec) / RegionDuration, 0.0f, 1.0f);
    Out.Coverage01 = CursorTrackerEvidenceCoverage01(Out.WindowEndSec - Out.WindowStartSec, PrerollSec, RegionDuration);

    if (!Frames || Frames->Num() == 0 || Out.WindowEndSec <= Out.WindowStartSec)
    {
        Out.PosteriorStartProgress01 = Out.PriorStartProgress01;
        Out.PosteriorEndProgress01 = Out.PriorEndProgress01;
        return Out;
    }

    float WeightedPhone = 0.0f;
    float WeightedBoundary = 0.0f;
    float WeightedSpeech = 0.0f;
    float WeightSum = 0.0f;
    float DensityMass = 0.0f;
    float LastTime = Out.WindowStartSec;

    for (const FOffgridAIStreamingAudioFeatureFrame& F : *Frames)
    {
        if (F.AudioBufferCenterSec < Out.WindowStartSec)
        {
            continue;
        }
        if (F.AudioBufferCenterSec > Out.WindowEndSec)
        {
            break;
        }

        const float FrameProgress = FMath::Clamp((F.AudioBufferCenterSec - RegionStartSec) / RegionDuration, 0.0f, 1.0f);
        const FOffgridAIExpectedPhone* Expected = FindExpectedPhoneNearRegionProgress(Plan, SentenceIndex, FrameProgress);
        float PhoneProb = 0.0f;
        if (Expected)
        {
            const EOffgridAIPhoneClass ExpectedClass = FOffgridAIOnlinePhoneAligner::ClassForPhoneBase(Expected->BasePhone);
            const FOffgridAIArticulatoryProbabilityField Field = FOffgridAIOnlinePhoneAligner::BuildArticulatoryProbabilityField(F);
            PhoneProb = FMath::Clamp(FOffgridAIOnlinePhoneAligner::ScoreForClass(Field.PhoneScores, ExpectedClass), 0.0f, 1.0f);
            const float BoundaryProb = FMath::Clamp(Field.Transition * (1.0f - Field.RedHerring), 0.0f, 1.0f);
            const float SpeechProb = FMath::Clamp(Field.Speech, 0.0f, 1.0f);
            Out.BestPhoneProbability = FMath::Max(Out.BestPhoneProbability, PhoneProb);
            Out.BestBoundaryProbability = FMath::Max(Out.BestBoundaryProbability, BoundaryProb);
            const float Step = FMath::Clamp(F.AudioBufferCenterSec - LastTime, 0.006f, 0.030f);
            const float Density = ProgressEstimatorLocalDensity(Frames, F.AudioBufferCenterSec);
            DensityMass += Density * Step;
            WeightedPhone += PhoneProb;
            WeightedBoundary += BoundaryProb;
            WeightedSpeech += SpeechProb;
            WeightSum += 1.0f;
            LastTime = F.AudioBufferCenterSec;
        }
    }

    if (WeightSum > 0.0f)
    {
        Out.MeanPhoneProbability = WeightedPhone / WeightSum;
        Out.MeanBoundaryProbability = WeightedBoundary / WeightSum;
        Out.MeanSpeechProbability = WeightedSpeech / WeightSum;
    }

    const float DensityProgressAtWindowStart = AudioProgressDensityProgress01(Frames, RegionStartSec, RegionEndSec, Out.WindowStartSec);
    const float DensityProgressAtWindowEnd = AudioProgressDensityProgress01(Frames, RegionStartSec, RegionEndSec, Out.WindowEndSec);
    Out.PosteriorStartProgress01 = FMath::Clamp(0.62f * Out.PriorStartProgress01 + 0.38f * DensityProgressAtWindowStart, 0.0f, 1.0f);
    Out.PosteriorEndProgress01 = FMath::Clamp(0.62f * Out.PriorEndProgress01 + 0.38f * DensityProgressAtWindowEnd, 0.0f, 1.0f);

    const float WindowHasUsefulAudio = FMath::Clamp((DensityMass / FMath::Max(Out.WindowEndSec - Out.WindowStartSec, 0.001f)) / 1.25f, 0.0f, 1.0f);
    const float ProbabilityAgreement = FMath::Clamp(0.55f * Out.MeanPhoneProbability + 0.25f * Out.BestPhoneProbability + 0.20f * Out.MeanSpeechProbability, 0.0f, 1.0f);
    Out.MatchConfidence = FMath::Clamp(WindowHasUsefulAudio * ProbabilityAgreement * Out.Coverage01, 0.0f, 1.0f);

    EvaluatePrerollWindowSequenceMatch(Plan, Frames, SentenceIndex, RegionStartSec, RegionEndSec, Out.WindowStartSec, Out.WindowEndSec,
        bHasPreviousCursor, PreviousCursorSentenceIndex, PreviousCursorProgress01, PreviousCursorPlaybackSec, Out);
    return Out;
}

static void BuildAudioProgressPriorRegionWordSpans(
    const FOffgridAITextVisemePlan& Plan,
    const TArray<FOffgridAIStreamingSpeechIsland>* Islands,
    const TArray<FOffgridAIStreamingAudioFeatureFrame>* /*Frames*/,
    int32 SentenceIndex,
    float ObservedAudioEndSec,
    int32 SentenceCount,
    TMap<int32, FOffgridAIRegionWordSpan>& OutSpans)
{
    // Standalone/shim-safe fallback used when the audio progress curve cannot be
    // sampled. It keeps the advisory progress system deterministic without
    // reintroducing a word-identity experiment: words are simply laid out across
    // the observed speech region according to CMU-derived expected phone weight.
    float RegionStart = 0.0f;
    float RegionEnd = 0.0f;
    if (!TryGetObservedSpeechRegionForSentence(Islands, SentenceCount, SentenceIndex, ObservedAudioEndSec, RegionStart, RegionEnd))
    {
        return;
    }

    TArray<int32> Words;
    for (int32 W = 0; W < Plan.WordSentenceIslandIndices.Num(); ++W)
    {
        if (Plan.WordSentenceIslandIndices[W] == SentenceIndex)
        {
            Words.Add(W);
        }
    }
    if (Words.Num() == 0)
    {
        return;
    }

    float TotalWeight = 0.0f;
    for (int32 WordIndex : Words)
    {
        TotalWeight += WordExpectedWeight(Plan, WordIndex);
    }
    TotalWeight = FMath::Max(TotalWeight, 0.001f);

    const float RegionDuration = FMath::Max(RegionEnd - RegionStart, 0.001f);
    float Cursor = RegionStart;
    for (int32 I = 0; I < Words.Num(); ++I)
    {
        const int32 WordIndex = Words[I];
        const float Fraction = WordExpectedWeight(Plan, WordIndex) / TotalWeight;
        const float Next = (I == Words.Num() - 1) ? RegionEnd : Cursor + RegionDuration * Fraction;

        FOffgridAIRegionWordSpan Span;
        Span.WordIndex = WordIndex;
        Span.StartSec = FMath::Clamp(Cursor, RegionStart, RegionEnd);
        Span.EndSec = FMath::Clamp(Next, RegionStart, RegionEnd);
        if (Span.EndSec > Span.StartSec + 0.012f)
        {
            OutSpans.Add(Span.WordIndex, Span);
        }
        Cursor = Next;
    }
}

static float ProgressEstimatorLocalDensity(
    const TArray<FOffgridAIStreamingAudioFeatureFrame>* Frames,
    float TimeSec)
{
    // Continuous transcript-progress model. The hidden state is not a discrete
    // "current word"; it is progress along the transcript inside the active
    // speech region. Audio features modulate local progress velocity.
    if (!Frames || Frames->Num() == 0)
    {
        return 1.0f;
    }

    float BestDistance = TNumericLimits<float>::Max();
    FOffgridAIArticulatoryProbabilityField BestField;
    float LocalEnergy = 0.0f;
    bool bFoundFrame = false;

    for (const FOffgridAIStreamingAudioFeatureFrame& F : *Frames)
    {
        const float D = FMath::Abs(F.AudioBufferCenterSec - TimeSec);
        if (D < BestDistance)
        {
            BestDistance = D;
            BestField = FOffgridAIOnlinePhoneAligner::BuildArticulatoryProbabilityField(F);
            LocalEnergy = F.RMSNorm;
            bFoundFrame = true;
        }
        if (F.AudioBufferCenterSec > TimeSec + 0.090f)
        {
            break;
        }
    }

    if (!bFoundFrame)
    {
        return 1.0f;
    }

    const float Boundary = AudioProgressBoundaryConfidenceAround(Frames, TimeSec, 0.055f);

    // This is a velocity observation, not a boundary command.  All local acoustic
    // influences now come from the same articulatory probability field used by
    // phone scoring and lexical-transition hints.
    const float SpeechGate = 0.42f + 0.58f * BestField.Speech;
    const float EnergyDrive = FMath::Clamp(LocalEnergy * 0.22f, 0.0f, 0.22f);
    const float ArticulationDrive = FMath::Clamp(
        0.22f * BestField.Transition +
        0.16f * BestField.Release +
        0.12f * BestField.Closure +
        0.10f * BestField.SpectralChange +
        0.08f * BestField.EnergyChange,
        0.0f, 0.42f);

    return FMath::Clamp(SpeechGate * (1.0f + 0.62f * Boundary + EnergyDrive + ArticulationDrive), 0.16f, 2.90f);
}

static float ProgressEstimatorConfidenceAround(
    const TArray<FOffgridAIStreamingAudioFeatureFrame>* Frames,
    float TimeSec)
{
    if (!Frames || Frames->Num() == 0)
    {
        return 0.25f;
    }
    float Best = 0.0f;
    for (const FOffgridAIStreamingAudioFeatureFrame& F : *Frames)
    {
        if (F.AudioBufferCenterSec < TimeSec - 0.080f) continue;
        if (F.AudioBufferCenterSec > TimeSec + 0.080f) break;
        const FOffgridAIArticulatoryProbabilityField A = FOffgridAIOnlinePhoneAligner::BuildArticulatoryProbabilityField(F);
        const float Transition = AudioProgressBoundaryConfidenceAround(Frames, F.AudioBufferCenterSec, 0.050f);
        const float C = FMath::Clamp(0.50f * A.Speech + 0.28f * Transition + 0.22f * A.Transition, 0.0f, 1.0f);
        Best = FMath::Max(Best, C);
    }
    return Best;
}

static float ProgressEstimatorInterpolateCrossing(
    float T0,
    float P0,
    float T1,
    float P1,
    float TargetProgress)
{
    const float Denom = P1 - P0;
    if (FMath::Abs(Denom) < 0.000001f)
    {
        return 0.5f * (T0 + T1);
    }
    const float Alpha = FMath::Clamp((TargetProgress - P0) / Denom, 0.0f, 1.0f);
    return T0 + (T1 - T0) * Alpha;
}

static void BuildAudioProgressRegionWordSpans(
    const FOffgridAITextVisemePlan& Plan,
    const TArray<FOffgridAIStreamingSpeechIsland>* Islands,
    const TArray<FOffgridAIStreamingAudioFeatureFrame>* Frames,
    int32 SentenceIndex,
    float ObservedAudioEndSec,
    int32 SentenceCount,
    TMap<int32, FOffgridAIRegionWordSpan>& OutSpans)
{
    float RegionStart = 0.0f;
    float RegionEnd = 0.0f;
    if (!TryGetObservedSpeechRegionForSentence(Islands, SentenceCount, SentenceIndex, ObservedAudioEndSec, RegionStart, RegionEnd))
    {
        return;
    }

    TArray<int32> Words;
    for (int32 W = 0; W < Plan.WordSentenceIslandIndices.Num(); ++W)
    {
        if (Plan.WordSentenceIslandIndices[W] == SentenceIndex)
        {
            Words.Add(W);
        }
    }
    if (Words.Num() == 0)
    {
        return;
    }
    if (Words.Num() == 1)
    {
        FOffgridAIRegionWordSpan Span;
        Span.WordIndex = Words[0];
        Span.StartSec = RegionStart;
        Span.EndSec = RegionEnd;
        OutSpans.Add(Span.WordIndex, Span);
        return;
    }

    TArray<float> Weights;
    float TotalWeight = 0.0f;
    for (int32 WordIndex : Words)
    {
        const float Weight = WordExpectedWeight(Plan, WordIndex);
        Weights.Add(Weight);
        TotalWeight += Weight;
    }
    TotalWeight = FMath::Max(TotalWeight, 0.001f);

    const float RegionDuration = FMath::Max(RegionEnd - RegionStart, 0.001f);
    const float MinWordSpan = FMath::Clamp(RegionDuration / FMath::Max(static_cast<float>(Words.Num()) * 16.0f, 1.0f), 0.025f, 0.075f);

    // Build a monotonic continuous progress curve P(t).  P(RegionStart)=0 and
    // P(RegionEnd)=TotalWeight.  Audio features affect the local density of P,
    // while a uniform base density prevents missed acoustic boundaries from
    // freezing progress indefinitely.
    TArray<float> Times;
    Times.Add(RegionStart);
    if (Frames)
    {
        for (const FOffgridAIStreamingAudioFeatureFrame& F : *Frames)
        {
            if (F.AudioBufferCenterSec <= RegionStart + 0.002f)
            {
                continue;
            }
            if (F.AudioBufferCenterSec >= RegionEnd - 0.002f)
            {
                break;
            }
            Times.Add(F.AudioBufferCenterSec);
        }
    }
    Times.Add(RegionEnd);
    Times.Sort([](float A, float BVal) { return A < BVal; });

    // Deduplicate very-near analysis points.
    TArray<float> UniqueTimes;
    for (float T : Times)
    {
        if (UniqueTimes.Num() == 0 || FMath::Abs(T - UniqueTimes.Last()) > 0.004f)
        {
            UniqueTimes.Add(T);
        }
    }
    Times = UniqueTimes;
    if (Times.Num() < 2)
    {
        BuildAudioProgressPriorRegionWordSpans(Plan, Islands, Frames, SentenceIndex, ObservedAudioEndSec, SentenceCount, OutSpans);
        return;
    }

    TArray<float> RawProgress;
    RawProgress.SetNum(Times.Num());
    RawProgress[0] = 0.0f;
    for (int32 I = 1; I < Times.Num(); ++I)
    {
        const float PrevT = Times[I - 1];
        const float T = Times[I];
        const float Mid = 0.5f * (PrevT + T);
        const float Dt = FMath::Max(T - PrevT, 0.0001f);
        const float Density = ProgressEstimatorLocalDensity(Frames, Mid);
        RawProgress[I] = RawProgress[I - 1] + Density * Dt;
    }

    const float RawTotal = FMath::Max(RawProgress.Last(), 0.0001f);
    TArray<float> Progress;
    Progress.SetNum(RawProgress.Num());
    for (int32 I = 0; I < RawProgress.Num(); ++I)
    {
        const float AudioProgress = TotalWeight * (RawProgress[I] / RawTotal);
        const float PriorProgress = TotalWeight * ((Times[I] - RegionStart) / RegionDuration);
        // Conservative blend: prior supplies global stability; audio density bends
        // the clock toward real acoustic structure.
        Progress[I] = FMath::Clamp(0.62f * PriorProgress + 0.38f * AudioProgress, 0.0f, TotalWeight);
    }
    Progress[0] = 0.0f;
    Progress.Last() = TotalWeight;
    for (int32 I = 1; I < Progress.Num(); ++I)
    {
        Progress[I] = FMath::Max(Progress[I], Progress[I - 1] + 0.00001f);
    }
    Progress.Last() = TotalWeight;

    TArray<float> Boundaries;
    Boundaries.SetNum(Words.Num() + 1);
    Boundaries[0] = RegionStart;
    Boundaries.Last() = RegionEnd;

    float Target = 0.0f;
    int32 SearchIndex = 1;
    for (int32 B = 1; B < Words.Num(); ++B)
    {
        Target += Weights[B - 1];
        while (SearchIndex < Progress.Num() && Progress[SearchIndex] < Target)
        {
            ++SearchIndex;
        }
        if (SearchIndex <= 0 || SearchIndex >= Progress.Num())
        {
            Boundaries[B] = RegionStart + RegionDuration * (Target / TotalWeight);
        }
        else
        {
            Boundaries[B] = ProgressEstimatorInterpolateCrossing(
                Times[SearchIndex - 1], Progress[SearchIndex - 1],
                Times[SearchIndex], Progress[SearchIndex],
                Target);
        }

        // Softly snap toward a nearby strong acoustic transition, but never more
        // than the local prior can justify.  This preserves the progress model as
        // the primary state and treats boundary detections as observations.
        const float LocalConfidence = AudioProgressBoundaryConfidenceAround(Frames, Boundaries[B], 0.085f);
        if (LocalConfidence >= 0.30f && Frames)
        {
            float BestT = Boundaries[B];
            float BestScore = 0.0f;
            const float Radius = FMath::Clamp(RegionDuration * 0.055f, 0.055f, 0.180f);
            for (const FOffgridAIStreamingAudioFeatureFrame& F : *Frames)
            {
                if (F.AudioBufferCenterSec < Boundaries[B] - Radius) continue;
                if (F.AudioBufferCenterSec > Boundaries[B] + Radius) break;
                const float Score = AudioProgressBoundaryConfidenceAround(Frames, F.AudioBufferCenterSec, 0.045f);
                if (Score > BestScore)
                {
                    BestScore = Score;
                    BestT = F.AudioBufferCenterSec;
                }
            }
            const float SnapAlpha = FMath::Clamp(BestScore * 0.38f, 0.0f, 0.32f);
            Boundaries[B] = Boundaries[B] + (BestT - Boundaries[B]) * SnapAlpha;
        }
    }

    // Enforce a valid ordered segmentation.
    for (int32 I = 1; I < Boundaries.Num(); ++I)
    {
        Boundaries[I] = FMath::Max(Boundaries[I], Boundaries[I - 1] + MinWordSpan);
    }
    for (int32 I = Boundaries.Num() - 2; I >= 0; --I)
    {
        Boundaries[I] = FMath::Min(Boundaries[I], Boundaries[I + 1] - MinWordSpan);
    }
    Boundaries[0] = RegionStart;
    Boundaries.Last() = RegionEnd;

    for (int32 I = 0; I < Words.Num(); ++I)
    {
        FOffgridAIRegionWordSpan Span;
        Span.WordIndex = Words[I];
        Span.StartSec = FMath::Clamp(Boundaries[I], RegionStart, RegionEnd);
        Span.EndSec = FMath::Clamp(Boundaries[I + 1], RegionStart, RegionEnd);
        if (Span.EndSec > Span.StartSec + 0.012f)
        {
            OutSpans.Add(Span.WordIndex, Span);
        }
    }
}


static float BoundaryConfidenceAround(
    const TArray<FOffgridAIStreamingAudioFeatureFrame>* Frames,
    float BoundarySec,
    float WindowSec)
{
    if (!Frames || Frames->Num() == 0 || BoundarySec < 0.0f)
    {
        return 0.0f;
    }

    const float WindowStart = BoundarySec - FMath::Max(WindowSec, 0.030f);
    const float WindowEnd = BoundarySec + FMath::Max(WindowSec, 0.030f);
    float Best = 0.0f;
    for (const FOffgridAIStreamingAudioFeatureFrame& F : *Frames)
    {
        if (F.AudioBufferCenterSec < WindowStart) continue;
        if (F.AudioBufferCenterSec > WindowEnd) break;
        const float Evidence = RuntimeSpeechEvidenceScore(F);
        const float LowEvidence = FMath::Clamp(1.0f - Evidence / 0.20f, 0.0f, 1.0f);
        const float Valley = F.bLocalRMSValley ? 0.32f : 0.0f;
        const float Flux = F.bLocalFluxPeak ? 0.24f : FMath::Clamp(F.Flux * 0.20f, 0.0f, 0.20f);
        const float VoicingShift = FMath::Clamp((1.0f - F.Periodicity) * F.RMSNorm * 0.22f, 0.0f, 0.22f);
        const float Score = FMath::Clamp(0.42f * LowEvidence + Valley + Flux + VoicingShift, 0.0f, 1.0f);
        Best = FMath::Max(Best, Score);
    }
    return Best;
}

static float DurationAdvancePrior(float ElapsedSec, float ExpectedSec)
{
    const float Expected = FMath::Max(ExpectedSec, 0.060f);
    const float Ratio = ElapsedSec / Expected;
    if (Ratio <= 0.35f) return 0.0f;
    if (Ratio <= 0.85f) return FMath::Clamp((Ratio - 0.35f) / 0.50f * 0.35f, 0.0f, 0.35f);
    if (Ratio <= 1.35f) return FMath::Clamp(0.35f + (Ratio - 0.85f) / 0.50f * 0.35f, 0.35f, 0.70f);
    return FMath::Clamp(0.70f + (Ratio - 1.35f) / 0.75f * 0.30f, 0.70f, 1.0f);
}

static void BuildAudioProgressMeasurements(
    const FOffgridAILipsyncRuntimeUpdateInput& Input,
    int32 UpdateOrdinal,
    TArray<FOffgridAIAudioProgressMeasurementRow>& OutRows)
{
    if (!Input.TextPlan || !Input.SpeechIslands)
    {
        return;
    }
    const FOffgridAITextVisemePlan& Plan = *Input.TextPlan;
    if (Plan.WordSentenceIslandIndices.Num() == 0)
    {
        return;
    }

    TArray<int32> SentenceIndices;
    // Progress diagnostics are speech-region diagnostics, not visible-viseme
    // diagnostics.  v8 built the sentence list from visible events, which meant
    // utterances with no selected visible events in a region produced
    // progress_rows=0 even though the transcript and speech region were valid.
    // Build from the word->sentence map instead so every text speech region can
    // be tracked.
    for (int32 SentenceIndex : Plan.WordSentenceIslandIndices)
    {
        if (SentenceIndex == INDEX_NONE) continue;
        bool bSeen = false;
        for (int32 Existing : SentenceIndices) { if (Existing == SentenceIndex) { bSeen = true; break; } }
        if (!bSeen) SentenceIndices.Add(SentenceIndex);
    }

    TMap<int32, FOffgridAIRegionWordSpan> WordSpans;
    for (int32 SentenceIndex : SentenceIndices)
    {
        BuildAudioProgressRegionWordSpans(Plan, Input.SpeechIslands, Input.AudioFeatureFrames, SentenceIndex,
            Input.ObservedAudioBufferEndSec, SentenceIndices.Num(), WordSpans);
    }

    int32 CurrentWord = INDEX_NONE;
    for (const auto& Pair : WordSpans)
    {
        const FOffgridAIRegionWordSpan& Span = Pair.second;
        if (Input.CurrentPlaybackSec >= Span.StartSec && Input.CurrentPlaybackSec <= Span.EndSec)
        {
            CurrentWord = Span.WordIndex;
            break;
        }
    }
    if (CurrentWord == INDEX_NONE)
    {
        float BestDistance = TNumericLimits<float>::Max();
        for (const auto& Pair : WordSpans)
        {
            const FOffgridAIRegionWordSpan& Span = Pair.second;
            const float D = Input.CurrentPlaybackSec < Span.StartSec ? Span.StartSec - Input.CurrentPlaybackSec : Input.CurrentPlaybackSec - Span.EndSec;
            if (D >= 0.0f && D < BestDistance)
            {
                BestDistance = D;
                CurrentWord = Span.WordIndex;
            }
        }
    }
    if (CurrentWord == INDEX_NONE)
    {
        return;
    }

    const FOffgridAIRegionWordSpan* Span = WordSpans.Find(CurrentWord);
    if (!Span)
    {
        return;
    }
    const int32 NextWord = CurrentWord + 1 < Plan.WordSentenceIslandIndices.Num() ? CurrentWord + 1 : INDEX_NONE;
    const float Expected = WordExpectedWeight(Plan, CurrentWord);
    const float Elapsed = FMath::Max(Input.CurrentPlaybackSec - Span->StartSec, 0.0f);
    const float DurationPrior = DurationAdvancePrior(Elapsed, Expected);
    const float BoundarySec = Span->EndSec;
    const float BoundaryConfidence = BoundaryConfidenceAround(Input.AudioFeatureFrames, BoundarySec, 0.075f);
    float BoundaryRedHerring = 0.0f;
    const float FilteredBoundaryConfidence = AudioProgressFilteredBoundaryConfidenceAround(Input.AudioFeatureFrames, BoundarySec, 0.075f, &BoundaryRedHerring);
    const float NextProb = FMath::Clamp(0.58f * FilteredBoundaryConfidence + 0.42f * DurationPrior, 0.0f, 1.0f);
    const float StillProb = FMath::Clamp(1.0f - NextProb, 0.0f, 1.0f);

    FName Reason = FName(TEXT("hold_current"));
    if (NextWord == INDEX_NONE)
    {
        Reason = FName(TEXT("last_word"));
    }
    else if (FilteredBoundaryConfidence >= 0.68f && DurationPrior >= 0.20f)
    {
        Reason = FName(TEXT("boundary_confidence"));
    }
    else if (DurationPrior >= 0.88f)
    {
        Reason = FName(TEXT("duration_prior_timeout"));
    }
    else if (FilteredBoundaryConfidence >= 0.82f)
    {
        Reason = FName(TEXT("strong_boundary_early"));
    }

    FOffgridAIAudioProgressMeasurementRow Row;
    Row.LineID = Input.LineID;
    Row.UpdateOrdinal = UpdateOrdinal;
    Row.CurrentPlaybackSec = Input.CurrentPlaybackSec;
    Row.ObservedAudioEndSec = Input.ObservedAudioBufferEndSec;
    Row.CurrentWordIndex = CurrentWord;
    for (const FOffgridAIExpectedPhone& P : Plan.ExpectedPhones)
    {
        if (P.WordIndex == CurrentWord && Row.CurrentWord.IsEmpty()) Row.CurrentWord = P.SourceWord;
        if (P.WordIndex == NextWord && Row.NextWord.IsEmpty()) Row.NextWord = P.SourceWord;
    }
    Row.NextWordIndex = NextWord;
    Row.WordStartSec = Span->StartSec;
    Row.WordEndSec = Span->EndSec;
    Row.ExpectedDurationSec = Expected;
    Row.ElapsedInWordSec = Elapsed;
    Row.DurationRatio = Elapsed / FMath::Max(Expected, 0.001f);
    Row.BoundaryTimeSec = BoundarySec;
    Row.BoundaryConfidence = BoundaryConfidence;
    Row.FilteredBoundaryConfidence = FilteredBoundaryConfidence;
    Row.BoundaryRedHerringProbability = BoundaryRedHerring;
    Row.DurationAdvancePrior = DurationPrior;

    // Diagnostic view of the audio-progress WIP: continuous transcript progress
    // within the current sentence/region.  This is weight-space progress, not a
    // hard word index.  Integer-ish crossings imply word transitions.
    float SentenceRegionStart = Span->StartSec;
    float SentenceRegionEnd = Span->EndSec;
    for (const auto& Pair : WordSpans)
    {
        const FOffgridAIRegionWordSpan& S = Pair.second;
        if (Plan.WordSentenceIslandIndices.IsValidIndex(S.WordIndex) &&
            Plan.WordSentenceIslandIndices[S.WordIndex] == Plan.WordSentenceIslandIndices[CurrentWord])
        {
            SentenceRegionStart = FMath::Min(SentenceRegionStart, S.StartSec);
            SentenceRegionEnd = FMath::Max(SentenceRegionEnd, S.EndSec);
        }
    }

    float CumulativeBefore = 0.0f;
    float TotalSentenceWeight = 0.0f;
    float CurrentWeight = FMath::Max(WordExpectedWeight(Plan, CurrentWord), 0.001f);
    int32 LocalWordOrdinal = 0;
    int32 CurrentLocalOrdinal = 0;
    for (int32 W = 0; W < Plan.WordSentenceIslandIndices.Num(); ++W)
    {
        if (!Plan.WordSentenceIslandIndices.IsValidIndex(W) ||
            !Plan.WordSentenceIslandIndices.IsValidIndex(CurrentWord) ||
            Plan.WordSentenceIslandIndices[W] != Plan.WordSentenceIslandIndices[CurrentWord])
        {
            continue;
        }
        const float WWeight = FMath::Max(WordExpectedWeight(Plan, W), 0.001f);
        if (W == CurrentWord)
        {
            CurrentLocalOrdinal = LocalWordOrdinal;
            CurrentWeight = WWeight;
        }
        if (W < CurrentWord)
        {
            CumulativeBefore += WWeight;
        }
        TotalSentenceWeight += WWeight;
        ++LocalWordOrdinal;
    }
    const float WordFraction = FMath::Clamp((Input.CurrentPlaybackSec - Span->StartSec) / FMath::Max(Span->EndSec - Span->StartSec, 0.001f), 0.0f, 1.0f);
    Row.TranscriptProgress = CumulativeBefore + WordFraction * CurrentWeight;
    Row.PriorTranscriptProgress = FMath::Clamp(TotalSentenceWeight * ((Input.CurrentPlaybackSec - SentenceRegionStart) / FMath::Max(SentenceRegionEnd - SentenceRegionStart, 0.001f)), 0.0f, FMath::Max(TotalSentenceWeight, 0.001f));
    Row.ProgressWordFloat = static_cast<float>(CurrentLocalOrdinal) + WordFraction;

    Row.RegionStartSec = SentenceRegionStart;
    Row.RegionEndSec = SentenceRegionEnd;
    Row.RegionProgress01 = FMath::Clamp(Row.TranscriptProgress / FMath::Max(TotalSentenceWeight, 0.001f), 0.0f, 1.0f);
    Row.RegionPriorProgress01 = FMath::Clamp(Row.PriorTranscriptProgress / FMath::Max(TotalSentenceWeight, 0.001f), 0.0f, 1.0f);

    const int32 CurrentSentenceIndex = Plan.WordSentenceIslandIndices.IsValidIndex(CurrentWord)
        ? Plan.WordSentenceIslandIndices[CurrentWord]
        : 0;
    const FOffgridAIPrerollWindowLandscape Window = BuildPrerollWindowLandscape(
        Plan, Input.AudioFeatureFrames, CurrentSentenceIndex, SentenceRegionStart, SentenceRegionEnd,
        Input.CurrentPlaybackSec, Input.PrerollSec, Input.ObservedAudioBufferEndSec,
        Input.bHasPreviousCursor, Input.PreviousCursorSentenceIndex, Input.PreviousCursorProgress01, Input.PreviousCursorPlaybackSec);
    Row.PrerollWindowStartSec = Window.WindowStartSec;
    Row.PrerollWindowEndSec = Window.WindowEndSec;
    Row.PrerollWindowDurationSec = FMath::Max(Window.WindowEndSec - Window.WindowStartSec, 0.0f);
    Row.PrerollWindowCoverage01 = Window.Coverage01;
    Row.PrerollWindowRegionStartProgress01 = Window.PriorStartProgress01;
    Row.PrerollWindowRegionEndProgress01 = Window.PriorEndProgress01;
    Row.PrerollWindowPosteriorStartProgress01 = Window.PosteriorStartProgress01;
    Row.PrerollWindowPosteriorEndProgress01 = Window.PosteriorEndProgress01;
    Row.PrerollWindowBestPhoneProbability = Window.BestPhoneProbability;
    Row.PrerollWindowMeanPhoneProbability = Window.MeanPhoneProbability;
    Row.PrerollWindowBestBoundaryProbability = Window.BestBoundaryProbability;
    Row.PrerollWindowMeanBoundaryProbability = Window.MeanBoundaryProbability;
    Row.PrerollWindowMeanSpeechProbability = Window.MeanSpeechProbability;
    Row.PrerollWindowMatchConfidence = Window.MatchConfidence;
    Row.PrerollWindowSequenceMatchScore = Window.SequenceMatchScore;
    Row.PrerollWindowSequencePriorScore = Window.SequencePriorScore;
    Row.PrerollWindowSequenceScoreGap = Window.SequenceScoreGap;
    Row.PrerollWindowSequenceStartProgress01 = Window.SequenceStartProgress01;
    Row.PrerollWindowSequenceEndProgress01 = Window.SequenceEndProgress01;
    Row.PrerollWindowSequenceOffsetMs = Window.SequenceOffsetMs;
    Row.PrerollWindowPosteriorMeanProgress01 = Window.PosteriorMeanProgress01;
    Row.PrerollWindowPosteriorStdDevProgress01 = Window.PosteriorStdDevProgress01;
    Row.PrerollWindowPosteriorPeakProgress01 = Window.PosteriorPeakProgress01;
    Row.PrerollWindowPosteriorPeakProbability = Window.PosteriorPeakProbability;
    Row.PrerollWindowPosteriorEntropy01 = Window.PosteriorEntropy01;
    Row.PrerollWindowTrackedPriorProgress01 = Input.bHasPreviousCursor && Input.PreviousCursorSentenceIndex == CurrentSentenceIndex
        ? FMath::Clamp(Input.PreviousCursorProgress01 + (Input.CurrentPlaybackSec - Input.PreviousCursorPlaybackSec) / FMath::Max(SentenceRegionEnd - SentenceRegionStart, 0.001f), 0.0f, 1.0f)
        : Row.PrerollWindowRegionStartProgress01;
    Row.PrerollWindowExpectedStreamScore = Window.ExpectedStreamScore;
    Row.PrerollWindowPosteriorPeakPhoneIndex = Window.PosteriorPeakPhoneIndex;
    Row.PrerollWindowCommitSafeEndSec = Window.CommitSafeEndSec;

    // v11: single-source advisory progress.  Earlier patches blended time,
    // density, boundary, and phone heuristics into a separate progress estimate.
    // Those experiments are no longer competing systems.  They remain logged as
    // diagnostic features, but the hidden state is now the speech-region cursor
    // produced by the rolling preroll-window tracker above.
    Row.TimePriorProgress01 = FMath::Clamp((Input.CurrentPlaybackSec - SentenceRegionStart) / FMath::Max(SentenceRegionEnd - SentenceRegionStart, 0.001f), 0.0f, 1.0f);
    Row.AudioDensityProgress01 = AudioProgressDensityProgress01(Input.AudioFeatureFrames, SentenceRegionStart, SentenceRegionEnd, Input.CurrentPlaybackSec);

    float BoundaryWeightedProgress = 0.0f;
    float BoundaryWeightSum = 0.0f;
    for (const auto& Pair : WordSpans)
    {
        const FOffgridAIRegionWordSpan& S = Pair.second;
        if (S.WordIndex == CurrentWord)
        {
            continue;
        }
        if (!Plan.WordSentenceIslandIndices.IsValidIndex(S.WordIndex) ||
            !Plan.WordSentenceIslandIndices.IsValidIndex(CurrentWord) ||
            Plan.WordSentenceIslandIndices[S.WordIndex] != Plan.WordSentenceIslandIndices[CurrentWord])
        {
            continue;
        }
        if (S.EndSec <= SentenceRegionStart || S.EndSec >= SentenceRegionEnd)
        {
            continue;
        }
        float Red = 0.0f;
        const float Conf = AudioProgressFilteredBoundaryConfidenceAround(Input.AudioFeatureFrames, S.EndSec, 0.075f, &Red);
        const float TemporalGate = S.EndSec <= Input.CurrentPlaybackSec
            ? 1.0f
            : FMath::Clamp(1.0f - (S.EndSec - Input.CurrentPlaybackSec) / 0.180f, 0.0f, 1.0f);
        const float W = FMath::Clamp(Conf * TemporalGate, 0.0f, 1.0f);
        BoundaryWeightedProgress += W * FMath::Clamp((S.EndSec - SentenceRegionStart) / FMath::Max(SentenceRegionEnd - SentenceRegionStart, 0.001f), 0.0f, 1.0f);
        BoundaryWeightSum += W;
    }
    Row.BoundaryEvidenceProgress01 = BoundaryWeightSum > 0.001f
        ? FMath::Clamp(BoundaryWeightedProgress / BoundaryWeightSum, 0.0f, 1.0f)
        : Row.TimePriorProgress01;
    Row.PhoneExpectationProgress01 = AudioProgressPhoneExpectationProgress01(Plan, Input.AudioFeatureFrames,
        Plan.WordSentenceIslandIndices.IsValidIndex(CurrentWord) ? Plan.WordSentenceIslandIndices[CurrentWord] : 0,
        SentenceRegionStart, SentenceRegionEnd, Input.CurrentPlaybackSec);

    // v14 pivot: do not let the weak 350 ms cursor posterior replace the timing
    // prior.  The transcript plan remains the known path; the cursor, density,
    // boundary, and phone evidence are diagnostic/anchor observations used to
    // estimate retrospective drift and advisory future rate correction.
    Row.EstimatedRegionProgress01 = Row.TimePriorProgress01;
    Row.RegionProgress01 = Row.EstimatedRegionProgress01;
    Row.AnimationProgress01 = Row.TimePriorProgress01;
    Row.ProgressError01 = Row.PrerollWindowPosteriorMeanProgress01 - Row.TimePriorProgress01;

    const FOffgridAIMicroPauseTimingDiagnostics MicroPauses = AnalyzeMicroPausesInRegion(
        Input.AudioFeatureFrames, SentenceRegionStart, SentenceRegionEnd, Input.CurrentPlaybackSec);
    Row.MicroPauseCount50ms = MicroPauses.Count50ms;
    Row.MicroPauseCount75ms = MicroPauses.Count75ms;
    Row.MicroPauseCount120ms = MicroPauses.Count120ms;
    Row.NearestMicroPauseTimeSec = MicroPauses.NearestCenterSec;
    Row.NearestMicroPauseDurationSec = MicroPauses.NearestDurationSec;
    Row.NearestMicroPauseProgress01 = MicroPauses.NearestProgress01;

    // Retrospective drift: after some audio has passed, compare independent
    // anchor-style observations against where the prior thought we were.  This
    // is intentionally backward-looking; it should bend future play rate rather
    // than rewriting already-played visemes.
    const float RegionDurationForCursor = FMath::Max(SentenceRegionEnd - SentenceRegionStart, 0.001f);
    const float CursorConfForDrift = FMath::Clamp(
        Row.PrerollWindowPosteriorPeakProbability * (1.0f - Row.PrerollWindowPosteriorEntropy01) * Row.PrerollWindowCoverage01,
        0.0f, 1.0f);
    const float BoundaryConfForDrift = FMath::Clamp(Row.FilteredBoundaryConfidence * Row.DurationAdvancePrior, 0.0f, 1.0f);
    const float DensityConfForDrift = ProgressEstimatorConfidenceAround(Input.AudioFeatureFrames, Input.CurrentPlaybackSec);
    float DriftWeighted = 0.0f;
    float DriftWeight = 0.0f;

    auto AddDriftObservation = [&](float ObservedProgress, float Confidence)
    {
        const float C = FMath::Clamp(Confidence, 0.0f, 1.0f);
        if (C <= 0.001f)
        {
            return;
        }
        DriftWeighted += C * (FMath::Clamp(ObservedProgress, 0.0f, 1.0f) - Row.TimePriorProgress01);
        DriftWeight += C;
    };

    AddDriftObservation(Row.AudioDensityProgress01, 0.34f * DensityConfForDrift);
    AddDriftObservation(Row.BoundaryEvidenceProgress01, 0.44f * BoundaryConfForDrift);
    AddDriftObservation(Row.PhoneExpectationProgress01, 0.18f * Row.PrerollWindowMeanPhoneProbability);
    AddDriftObservation(Row.PrerollWindowPosteriorMeanProgress01, 0.22f * CursorConfForDrift);

    Row.RetrospectiveDrift01 = DriftWeight > 0.001f ? FMath::Clamp(DriftWeighted / DriftWeight, -0.35f, 0.35f) : 0.0f;
    Row.RetrospectiveDriftSec = Row.RetrospectiveDrift01 * RegionDurationForCursor;
    Row.RetrospectiveDriftConfidence = FMath::Clamp(DriftWeight, 0.0f, 1.0f);
    const float DriftCorrection = FMath::Clamp(Row.RetrospectiveDrift01 * 2.4f, -0.22f, 0.22f);
    Row.DriftSuggestedPlayRate = FMath::Clamp(1.0f + DriftCorrection * Row.RetrospectiveDriftConfidence, 0.86f, 1.16f);
    Row.FilteredPLLPlayRate = FMath::Clamp(Input.TimingPLLPlayRate, 0.90f, 1.10f);
    Row.FilteredPLLConfidence = FMath::Clamp(Input.TimingPLLConfidence, 0.0f, 1.0f);
    Row.PLLTempoRate = FMath::Clamp(Input.TimingPLLTempoRate, 0.90f, 1.10f);
    Row.PLLTempoConfidence = FMath::Clamp(Input.TimingPLLTempoConfidence, 0.0f, 1.0f);
    Row.PLLPhaseRate = FMath::Clamp(Input.TimingPLLPhaseRate, 0.90f, 1.10f);
    Row.TimingWarpRate = FMath::Clamp(Input.TimingWarpRate, 0.88f, 1.12f);
    Row.TimingWarpConfidence = FMath::Clamp(Input.TimingWarpConfidence, 0.0f, 1.0f);
    Row.TimingWarpAnchorCanonicalSec = Input.TimingWarpAnchorCanonicalSec;
    Row.TimingWarpAnchorObservedSec = Input.TimingWarpAnchorObservedSec;
    Row.TimingWarpErrorSec = Input.TimingWarpErrorSec;

    // Effective weights are kept for CSV compatibility.  v14 intentionally sets
    // them to zero because the row estimate is the timing prior; independent
    // evidence is logged as retrospective drift, not as a competing estimator.
    Row.EffectiveTimeWeight = 0.0f;
    Row.EffectiveDensityWeight = 0.0f;
    Row.EffectiveBoundaryWeight = 0.0f;
    Row.EffectivePhoneWeight = 0.0f;
    Row.EstimatorDisagreement01 = FMath::Max(
        FMath::Abs(Row.EstimatedRegionProgress01 - Row.TimePriorProgress01),
        FMath::Abs(Row.EstimatedRegionProgress01 - Row.AudioDensityProgress01));

    const float CursorConfidence = CursorConfForDrift;
    Row.EstimatedVelocity01PerSec = FMath::Clamp(1.0f / RegionDurationForCursor + Row.RetrospectiveDrift01 * Row.RetrospectiveDriftConfidence, 0.0f, 8.0f);
    Row.ProgressConfidence = Row.RetrospectiveDriftConfidence;

    Row.SuggestedPlayRate = Row.DriftSuggestedPlayRate;

    Row.StillCurrentProbability = StillProb;
    Row.NextWordProbability = NextProb;
    Row.bWouldAdvance = NextWord != INDEX_NONE && (Reason != FName(TEXT("hold_current")));
    Row.AdvanceReason = Reason;
    OutRows.Add(Row);
}

}

void FOffgridAILipsyncRuntimeSession::Reset()
{
    NPCID = NAME_None;
    LineID = NAME_None;
    DialogueText.Reset();
    PrerollSec = 0.350f;
    PlaybackSec = 0.0f;
    TimingPLLPlayRate = 1.0f;
    TimingPLLConfidence = 0.0f;
    TimingPLLTempoRate = 1.0f;
    TimingPLLTempoConfidence = 0.0f;
    TimingPLLPhaseRate = 1.0f;
    TimingWarpRate = 1.0f;
    TimingWarpConfidence = 0.0f;
    TimingWarpAnchorCanonicalSec = 0.0f;
    TimingWarpAnchorObservedSec = 0.0f;
    TimingWarpErrorSec = 0.0f;
    bBegun = false;
    bPlaybackStarted = false;
    bCommittedTrackBuilt = false;
    bInputStreamClosed = false;
    TextPlan = FOffgridAITextVisemePlan();
    Detector.Reset();
    ResolvedSpeechIslands.Reset();
    CommittedTrack = FOffgridAIAlignedVisemeTrack();
    AudioOccupancyDiagnosticRows.Reset();
    AudioProgressMeasurementRows.Reset();
    AudioOccupancyDiagnosticUpdateOrdinal = 0;
    StreamTailDiagnosticRow = FOffgridAIStreamTailDiagnosticRow();
    PCMChunkCount = 0;
    PCMBytesReceived = 0;
    PCMSamplesReceived = 0;
    LastPCMChunkSampleRate = 0;
    LastPCMChunkChannels = 0;
    LastPCMChunkStartSample = -1;
    LastPCMChunkEndSample = -1;
    bHaveLastCursor = false;
    LastCursorSentenceIndex = INDEX_NONE;
    LastCursorProgress01 = 0.0f;
    LastCursorPlaybackSec = 0.0f;
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
    Input.TimingPLLPlayRate = TimingPLLPlayRate;
    Input.TimingPLLConfidence = TimingPLLConfidence;
    Input.TimingPLLTempoRate = TimingPLLTempoRate;
    Input.TimingPLLTempoConfidence = TimingPLLTempoConfidence;
    Input.TimingPLLPhaseRate = TimingPLLPhaseRate;
    Input.TimingWarpRate = TimingWarpRate;
    Input.TimingWarpConfidence = TimingWarpConfidence;
    Input.TimingWarpAnchorCanonicalSec = TimingWarpAnchorCanonicalSec;
    Input.TimingWarpAnchorObservedSec = TimingWarpAnchorObservedSec;
    Input.TimingWarpErrorSec = TimingWarpErrorSec;
    Input.ObservedAudioBufferEndSec = Detector.GetObservedAudioBufferEndSec();
    Input.bInputStreamClosed = bInputStreamClosed;
    Input.NPCID = NPCID;
    Input.LineID = LineID;
    FOffgridAILipsyncRuntimeAdapter::UpdateCommittedTrack(Input, CommittedTrack, bCommittedTrackBuilt);
    RecordAudioProgressMeasurements(PlaybackSec);
    RecordAlignmentDiagnostics(PlaybackSec, false);
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
    Input.TimingPLLPlayRate = TimingPLLPlayRate;
    Input.TimingPLLConfidence = TimingPLLConfidence;
    Input.TimingPLLTempoRate = TimingPLLTempoRate;
    Input.TimingPLLTempoConfidence = TimingPLLTempoConfidence;
    Input.TimingPLLPhaseRate = TimingPLLPhaseRate;
    Input.TimingWarpRate = TimingWarpRate;
    Input.TimingWarpConfidence = TimingWarpConfidence;
    Input.TimingWarpAnchorCanonicalSec = TimingWarpAnchorCanonicalSec;
    Input.TimingWarpAnchorObservedSec = TimingWarpAnchorObservedSec;
    Input.TimingWarpErrorSec = TimingWarpErrorSec;
    Input.ObservedAudioBufferEndSec = Detector.GetObservedAudioBufferEndSec();
    Input.bInputStreamClosed = true;
    Input.bPlaybackFinalized = true; // Final drain: represent every remaining planned viseme inside observed speech when possible.
    Input.NPCID = NPCID;
    Input.LineID = LineID;
    FOffgridAILipsyncRuntimeAdapter::UpdateCommittedTrack(Input, CommittedTrack, bCommittedTrackBuilt);
    RecordAudioProgressMeasurements(PlaybackSec);
    RecordAlignmentDiagnostics(PlaybackSec, true);
    bCommittedTrackBuilt = true;
}

void FOffgridAILipsyncRuntimeSession::UpdatePlaybackGate(float ObservedEndSec)
{
    if (!bPlaybackStarted && ObservedEndSec >= FMath::Max(PrerollSec, 0.0f))
    {
        bPlaybackStarted = true;
    }
}

void FOffgridAILipsyncRuntimeSession::RefreshResolvedSpeechIslands()
{
    ResolvedSpeechIslands = ResolveSpeechIslands(
        TextPlan,
        Detector.GetIslands(),
        Detector.GetFeatureFrames(),
        Detector.GetObservedAudioBufferEndSec());
}


void FOffgridAILipsyncRuntimeSession::RecordAudioProgressMeasurements(float CurrentPlaybackSec)
{
    if (!bBegun)
    {
        return;
    }

    FOffgridAILipsyncRuntimeUpdateInput Input;
    Input.TextPlan = &TextPlan;
    Input.SpeechIslands = &ResolvedSpeechIslands;
    Input.AudioFeatureFrames = &Detector.GetFeatureFrames();
    Input.CurrentPlaybackSec = FMath::Max(CurrentPlaybackSec, 0.0f);
    Input.PrerollSec = PrerollSec;
    Input.TimingPLLPlayRate = TimingPLLPlayRate;
    Input.TimingPLLConfidence = TimingPLLConfidence;
    Input.TimingPLLTempoRate = TimingPLLTempoRate;
    Input.TimingPLLTempoConfidence = TimingPLLTempoConfidence;
    Input.TimingPLLPhaseRate = TimingPLLPhaseRate;
    Input.TimingWarpRate = TimingWarpRate;
    Input.TimingWarpConfidence = TimingWarpConfidence;
    Input.TimingWarpAnchorCanonicalSec = TimingWarpAnchorCanonicalSec;
    Input.TimingWarpAnchorObservedSec = TimingWarpAnchorObservedSec;
    Input.TimingWarpErrorSec = TimingWarpErrorSec;
    Input.ObservedAudioBufferEndSec = Detector.GetObservedAudioBufferEndSec();
    Input.bInputStreamClosed = bInputStreamClosed;
    Input.bPlaybackFinalized = false;
    Input.NPCID = NPCID;
    Input.LineID = LineID;
    Input.bHasPreviousCursor = bHaveLastCursor;
    Input.PreviousCursorSentenceIndex = LastCursorSentenceIndex;
    Input.PreviousCursorProgress01 = LastCursorProgress01;
    Input.PreviousCursorPlaybackSec = LastCursorPlaybackSec;

    const int32 OldRowCount = AudioProgressMeasurementRows.Num();
    BuildAudioProgressMeasurements(Input, AudioOccupancyDiagnosticUpdateOrdinal, AudioProgressMeasurementRows);
    if (AudioProgressMeasurementRows.Num() > OldRowCount)
    {
        const FOffgridAIAudioProgressMeasurementRow& R = AudioProgressMeasurementRows.Last();
        if (R.RegionEndSec > R.RegionStartSec && R.PrerollWindowCoverage01 > 0.05f)
        {
            bHaveLastCursor = true;
            LastCursorProgress01 = R.PrerollWindowPosteriorMeanProgress01;
            LastCursorPlaybackSec = R.CurrentPlaybackSec;
            LastCursorSentenceIndex = TextPlan.WordSentenceIslandIndices.IsValidIndex(R.CurrentWordIndex)
                ? TextPlan.WordSentenceIslandIndices[R.CurrentWordIndex]
                : INDEX_NONE;
        }

        // v17 monotonic time warp.  We already know the ordered CMU path;
        // the remaining unknown is the mapping from canonical transcript time
        // to observed audio time.  Each trustworthy retrospective drift sample
        // becomes a soft knot in that mapping.  The derived warp rate is then
        // passed forward as a duration prior for future, uncommitted visemes.
        const float TargetConfidence = FMath::Clamp(R.RetrospectiveDriftConfidence, 0.0f, 1.0f);
        const float LocalPhaseRate = FMath::Clamp(R.DriftSuggestedPlayRate, 0.88f, 1.12f);
        const float ObservedTempoRate = FMath::Clamp(LocalPhaseRate, 0.90f, 1.10f);

        if (TargetConfidence > 0.08f && R.RegionEndSec > R.RegionStartSec)
        {
            const float RegionDuration = FMath::Max(R.RegionEndSec - R.RegionStartSec, 0.001f);
            const float CanonicalProgress = FMath::Clamp(R.TimePriorProgress01, 0.0f, 1.0f);
            const float ObservedProgress = FMath::Clamp(CanonicalProgress + R.RetrospectiveDrift01, 0.0f, 1.0f);
            const float CanonicalAnchorSec = R.RegionStartSec + CanonicalProgress * RegionDuration;
            const float ObservedAnchorSec = R.RegionStartSec + ObservedProgress * RegionDuration;
            const float AnchorErrorSec = ObservedAnchorSec - CanonicalAnchorSec;

            const float WarpAlpha = FMath::Clamp(0.030f + 0.180f * TargetConfidence, 0.0f, 0.210f);
            TimingWarpRate = FMath::Clamp(
                TimingWarpRate * (1.0f - WarpAlpha) + ObservedTempoRate * WarpAlpha,
                0.90f, 1.10f);
            TimingWarpConfidence = FMath::Clamp(
                TimingWarpConfidence * 0.975f + TargetConfidence * 0.085f,
                0.0f, 1.0f);
            TimingWarpAnchorCanonicalSec = CanonicalAnchorSec;
            TimingWarpAnchorObservedSec = ObservedAnchorSec;
            TimingWarpErrorSec = AnchorErrorSec;

            // Compatibility fields still expose the old PLL vocabulary, but
            // they now mirror the warp model rather than acting as a competing
            // controller.
            TimingPLLTempoRate = TimingWarpRate;
            TimingPLLTempoConfidence = TimingWarpConfidence;
            TimingPLLPhaseRate = FMath::Clamp(
                TimingPLLPhaseRate * (1.0f - WarpAlpha) + LocalPhaseRate * WarpAlpha,
                0.88f, 1.12f);
        }
        else
        {
            TimingWarpConfidence = FMath::Clamp(TimingWarpConfidence * 0.990f, 0.0f, 1.0f);
            TimingPLLTempoConfidence = TimingWarpConfidence;
            TimingPLLPhaseRate = FMath::Clamp(TimingPLLPhaseRate * 0.970f + 1.0f * 0.030f, 0.88f, 1.12f);
        }

        const float WarpBlend = FMath::Clamp(0.25f + 0.55f * TimingWarpConfidence, 0.0f, 0.80f);
        const float PhaseBlend = FMath::Clamp(0.08f + 0.22f * TargetConfidence, 0.0f, 0.30f);
        const float WarpBackboneRate = 1.0f * (1.0f - WarpBlend) + TimingWarpRate * WarpBlend;
        const float CombinedRate = WarpBackboneRate * (1.0f - PhaseBlend) + TimingPLLPhaseRate * PhaseBlend;

        const float SevereDrift = FMath::Abs(R.RetrospectiveDrift01);
        const float MinRate = SevereDrift > 0.22f ? 0.88f : 0.92f;
        const float MaxRate = SevereDrift > 0.22f ? 1.12f : 1.08f;
        TimingPLLPlayRate = FMath::Clamp(CombinedRate, MinRate, MaxRate);
        TimingPLLConfidence = FMath::Clamp(
            TimingWarpConfidence * 0.80f + TargetConfidence * 0.20f,
            0.0f, 1.0f);
    }
}

void FOffgridAILipsyncRuntimeSession::RecordAlignmentDiagnostics(float CurrentPlaybackSec, bool bFinalReplay)
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
        R.AlignedPhoneStartSec = E.AlignedPhoneStartSeconds;
        R.AlignedPhoneEndSec = E.AlignedPhoneEndSeconds;
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
    if (!Input.TextPlan)
    {
        return;
    }

    const FOffgridAITextVisemePlan& Plan = *Input.TextPlan;
    InOutTrack.NPCID = Input.NPCID;
    InOutTrack.LineID = Input.LineID;
    if (InOutTrack.RuntimeFirstAlignedObservedEndSeconds.Num() != Plan.ExpectedPhones.Num())
    {
        InOutTrack.RuntimeFirstAlignedObservedEndSeconds.Init(-1.0f, Plan.ExpectedPhones.Num());
    }
    if (InOutTrack.RuntimeObservedPhoneStartSeconds.Num() != Plan.ExpectedPhones.Num())
    {
        InOutTrack.RuntimeObservedPhoneStartSeconds.Init(-1.0f, Plan.ExpectedPhones.Num());
        InOutTrack.RuntimeObservedPhoneEndSeconds.Init(-1.0f, Plan.ExpectedPhones.Num());
    }
    if (Plan.Events.Num() == 0)
    {
        bInOutTrackBuilt = true;
        return;
    }

    FOffgridAIOnlinePhoneAlignmentInput AlignmentInput;
    AlignmentInput.Plan = &Plan;
    AlignmentInput.AudioFeatureFrames = Input.AudioFeatureFrames;
    AlignmentInput.SpeechIslands = Input.SpeechIslands;
    AlignmentInput.ObservedAudioEndSec = Input.ObservedAudioBufferEndSec;
    AlignmentInput.PlaybackSec = Input.CurrentPlaybackSec;
    AlignmentInput.LookaheadSec = FMath::Max(Input.PrerollSec, 0.0f);
    AlignmentInput.CommitLagSec = 0.120f;
    AlignmentInput.TimingPLLPlayRate = Input.TimingPLLPlayRate;
    AlignmentInput.TimingPLLConfidence = Input.TimingPLLConfidence;
    AlignmentInput.TimingWarpRate = Input.TimingWarpRate;
    AlignmentInput.TimingWarpConfidence = Input.TimingWarpConfidence;
    // Input-stream close means all audio evidence for this line is known.
    // Final drain is allowed to use transcript-duration fallback so weak or
    // under-resolved acoustic evidence never suppresses planned visemes.
    AlignmentInput.bFinal = Input.bInputStreamClosed || Input.bPlaybackFinalized;
    const FOffgridAIOnlinePhoneAlignmentResult Alignment = FOffgridAIOnlinePhoneAligner::Compute(AlignmentInput);

    for (int32 PhoneIndex = 0; PhoneIndex < Plan.ExpectedPhones.Num(); ++PhoneIndex)
    {
        if (!Alignment.PhoneStartSeconds.IsValidIndex(PhoneIndex)
            || !Alignment.PhoneEndSeconds.IsValidIndex(PhoneIndex)
            || Alignment.PhoneEndSeconds[PhoneIndex] <= Alignment.PhoneStartSeconds[PhoneIndex])
        {
            continue;
        }

        const FName AdvanceReason = Alignment.PhoneAdvanceReasons.IsValidIndex(PhoneIndex)
            ? Alignment.PhoneAdvanceReasons[PhoneIndex]
            : NAME_None;
        const float PhoneStart = Alignment.PhoneStartSeconds[PhoneIndex];
        const float PhoneEnd = Alignment.PhoneEndSeconds[PhoneIndex];
        const bool bAcousticPhoneEvidence = AdvanceReason != FName(TEXT("final_duration_drain"))
            && PhoneSpanOverlapsObservedSpeech(Input.SpeechIslands, PhoneStart, PhoneEnd, 0.120f);
        if (bAcousticPhoneEvidence)
        {
            InOutTrack.RuntimeObservedPhoneStartSeconds[PhoneIndex] = PhoneStart;
            InOutTrack.RuntimeObservedPhoneEndSeconds[PhoneIndex] = PhoneEnd;
        }
    }

    const int32 FirstNewIndex = FirstUncommittedEventIndex(InOutTrack);
    float LastCenter = LastCommittedCenter(InOutTrack);

    // Bounded commit lead. Late sentence events remain provisional until playback
    // is near them. Once the TTS stream closes, all future audio is technically
    // observed, but committing the full tail immediately makes conversational
    // endings rigid and prevents later ticks from using fresher playback-local
    // context. Keep final drain streaming-like; at actual playback finalization
    // allow a full drain.
    constexpr float MaxStreamingCommitLeadSec = 0.800f;
    const float PlaybackLeadHorizon = Input.CurrentPlaybackSec + MaxStreamingCommitLeadSec;
    const float PrerollHorizon = Input.CurrentPlaybackSec + FMath::Max(Input.PrerollSec, 0.0f);
    const float RawObservedHorizon = (Input.bInputStreamClosed || Input.bPlaybackFinalized)
        ? FMath::Max(PrerollHorizon, Input.ObservedAudioBufferEndSec + 0.250f)
        : PrerollHorizon;
    const float CommitTimeHorizon = Input.bPlaybackFinalized
        ? RawObservedHorizon
        : FMath::Min(RawObservedHorizon, PlaybackLeadHorizon);
    const float FirstSpeechStart = FirstConfirmedSpeechStart(Input.SpeechIslands);
    if (InOutTrack.RuntimeFirstAlignedObservedEndSeconds.Num() < Plan.ExpectedPhones.Num())
    {
        const int32 OldNum = InOutTrack.RuntimeFirstAlignedObservedEndSeconds.Num();
        InOutTrack.RuntimeFirstAlignedObservedEndSeconds.SetNum(Plan.ExpectedPhones.Num());
        for (int32 Index = OldNum; Index < InOutTrack.RuntimeFirstAlignedObservedEndSeconds.Num(); ++Index)
        {
            InOutTrack.RuntimeFirstAlignedObservedEndSeconds[Index] = -1.0f;
        }
    }

    for (int32 EventIndex = FirstNewIndex; EventIndex < Plan.Events.Num(); ++EventIndex)
    {
        const FOffgridAITextVisemeEvent& T = Plan.Events[EventIndex];
        const int32 PhoneIndex = FOffgridAIOnlinePhoneAligner::FindPhoneForEvent(Plan, T);
        const bool bRawPhoneEvidence = PhoneIndex != INDEX_NONE && HasVisibleAlignedPhone(Alignment, PhoneIndex);
        const float EvidencePhoneStart = (bRawPhoneEvidence && Alignment.PhoneStartSeconds.IsValidIndex(PhoneIndex)) ? Alignment.PhoneStartSeconds[PhoneIndex] : -1.0f;
        const float EvidencePhoneEnd = (bRawPhoneEvidence && Alignment.PhoneEndSeconds.IsValidIndex(PhoneIndex)) ? Alignment.PhoneEndSeconds[PhoneIndex] : -1.0f;
        const FName EvidenceAdvanceReason = (bRawPhoneEvidence && Alignment.PhoneAdvanceReasons.IsValidIndex(PhoneIndex)) ? Alignment.PhoneAdvanceReasons[PhoneIndex] : NAME_None;
        const bool bPhoneHasAcousticOwnership = bRawPhoneEvidence
            && EvidenceAdvanceReason != FName(TEXT("final_duration_drain"))
            && PhoneSpanOverlapsObservedSpeech(Input.SpeechIslands, EvidencePhoneStart, EvidencePhoneEnd, 0.120f);
        const bool bHasPhoneEvidence = bPhoneHasAcousticOwnership;
        if (bHasPhoneEvidence
            && InOutTrack.RuntimeFirstAlignedObservedEndSeconds.IsValidIndex(PhoneIndex)
            && InOutTrack.RuntimeFirstAlignedObservedEndSeconds[PhoneIndex] < 0.0f)
        {
            InOutTrack.RuntimeFirstAlignedObservedEndSeconds[PhoneIndex] = Input.ObservedAudioBufferEndSec;
        }

        const FName Pose = ResolvePose(T);
        const EOffgridAIPhoneClass PhoneClass = Plan.ExpectedPhones.IsValidIndex(PhoneIndex)
            ? FOffgridAIOnlinePhoneAligner::ClassForPhoneBase(Plan.ExpectedPhones[PhoneIndex].BasePhone)
            : EOffgridAIPhoneClass::Unknown;
        float Center = EventCenterNorm(T) * FMath::Max(Plan.EstimatedDurationSeconds, 0.001f);
        FName PlacementReason = FName(TEXT("duration_fallback"));

        if (bHasPhoneEvidence)
        {
            const float PhoneStart = Alignment.PhoneStartSeconds[PhoneIndex];
            const float PhoneEnd = Alignment.PhoneEndSeconds[PhoneIndex];
            const float PhoneCenter = CenterForAlignedPhone(Pose, PhoneClass, PhoneStart, PhoneEnd);
            Center = PhoneCenter;
            PlacementReason = FName(TEXT("streaming_forced_alignment"));
        }

        // The first visible state must not pre-open before confirmed speech.
        // Render windows can begin before their center, so protect the first
        // event by requiring its center to land far enough inside the first
        // confirmed speech span.
        const float EventSpan = SpanForPose(Pose);
        float BoundaryGuardMinCenter = -1.0f;
        FName BoundaryGuardReason = NAME_None;
        float TailGuardMaxCenter = -1.0f;
        FName TailGuardReason = NAME_None;

        if (EventIndex == 0 && FirstSpeechStart >= 0.0f)
        {
            const float MinFirstCenter = FirstSpeechStart + FMath::Min(EventSpan * 0.42f, 0.070f);
            Center = FMath::Max(Center, MinFirstCenter);
        }

        // Boundary anchoring: the phone aligner is intentionally lightweight and
        // can place the first phone of a new phrase inside the previous acoustic
        // span when active speech time is compressed.  Do not let text-boundary
        // metadata cross a real detector gap.  This is still runtime-local
        // forced alignment: detector islands provide acoustic pause evidence, but
        // they do not own words or reorder events.
        if (EventIndex > 0 && InOutTrack.Events.Num() > 0)
        {
            const FOffgridAIAlignedVisemeEvent& PrevCommitted = InOutTrack.Events.Last();
            const bool bSentenceBoundary = T.SentenceIslandIndex != INDEX_NONE
                && PrevCommitted.SentenceIndex != INDEX_NONE
                && T.SentenceIslandIndex != PrevCommitted.SentenceIndex;
            const bool bPhraseBoundary = !bSentenceBoundary
                && T.PhraseIndex != INDEX_NONE
                && PrevCommitted.PhraseIndex != INDEX_NONE
                && T.PhraseIndex != PrevCommitted.PhraseIndex;

            if (bSentenceBoundary || bPhraseBoundary)
            {
                const TCHAR BoundaryPunctuation = BoundaryPunctuationBeforeEvent(Plan, EventIndex);
                const bool bSoftCommaBoundary = bPhraseBoundary && IsSoftCommaBoundary(BoundaryPunctuation);
                // Soft comma carry: do not convert ordinary short comma/list
                // gaps into hard phrase-start resets. A comma boundary only gets
                // a hard onset guard when the audio contains a clearly long
                // silence before the next item.
                const float RequiredGap = 0.075f; // v15: try 75 ms punctuation micro-pause anchors.
                const float MaxBoundarySearch = bSentenceBoundary ? 1.450f : (bSoftCommaBoundary ? 0.700f : 0.950f);
                float GapStart = -1.0f;
                float NextSpeechStart = -1.0f;
                const bool bFoundFeatureBoundary = FindNextFeatureSpeechOnsetAfter(
                    Input.AudioFeatureFrames,
                    PrevCommitted.FinalRenderCenterSeconds,
                    RequiredGap,
                    MaxBoundarySearch,
                    GapStart,
                    NextSpeechStart);

                if (!bFoundFeatureBoundary)
                {
                    const float IslandRequiredGap = 0.075f;
                    if (FindNextSpeechIslandStartAfter(Input.SpeechIslands, PrevCommitted.FinalRenderCenterSeconds, IslandRequiredGap, NextSpeechStart))
                    {
                        GapStart = PrevCommitted.FinalRenderCenterSeconds + IslandRequiredGap;
                    }
                }

                if (NextSpeechStart >= 0.0f)
                {
                    const float BoundaryLead = bSentenceBoundary ? 0.060f : 0.045f;
                    BoundaryGuardMinCenter = NextSpeechStart + FMath::Min(EventSpan * 0.38f, BoundaryLead);
                    BoundaryGuardReason = bSentenceBoundary
                        ? FName(TEXT("sentence_boundary_onset_guard"))
                        : FName(TEXT("phrase_boundary_onset_guard"));
                    Center = FMath::Max(Center, BoundaryGuardMinCenter);
                }
                else if (!Input.bInputStreamClosed && bSentenceBoundary)
                {
                    // For terminal punctuation, avoid pre-committing the next sentence
                    // until evidence for its speech span has actually arrived.
                    break;
                }
            }
        }

        // If this is the final visible event before a text boundary,
        // keep it inside the current acoustic phrase.  The global active-speech
        // aligner can otherwise stretch a final consonant/vowel into the silence
        // before the next sentence, which reads as an isolated mouth pop.
        if (Plan.Events.IsValidIndex(EventIndex + 1))
        {
            const FOffgridAITextVisemeEvent& NextTextEvent = Plan.Events[EventIndex + 1];
            const bool bNextSentenceBoundary = NextTextEvent.SentenceIslandIndex != INDEX_NONE
                && T.SentenceIslandIndex != INDEX_NONE
                && NextTextEvent.SentenceIslandIndex != T.SentenceIslandIndex;
            const bool bNextPhraseBoundary = !bNextSentenceBoundary
                && NextTextEvent.PhraseIndex != INDEX_NONE
                && T.PhraseIndex != INDEX_NONE
                && NextTextEvent.PhraseIndex != T.PhraseIndex;
            if (bNextSentenceBoundary || bNextPhraseBoundary)
            {
                const TCHAR NextBoundaryPunctuation = BoundaryPunctuationBeforeEvent(Plan, EventIndex + 1);
                const bool bNextSoftCommaBoundary = bNextPhraseBoundary && IsSoftCommaBoundary(NextBoundaryPunctuation);
                // Soft comma carry: keep phrase-tail protection for real
                // pauses, but do not clamp a list item's tail before every short
                // comma dip. This lets the forced path keep articulation
                // continuous through supported comma gaps.
                const float RequiredTailGap = 0.075f; // v15: try 75 ms punctuation micro-pause anchors.
                const float MaxTailSearch = bNextSentenceBoundary ? 1.350f : (bNextSoftCommaBoundary ? 0.700f : 0.750f);
                float TailGapStart = -1.0f;
                float TailNextSpeechStart = -1.0f;
                if (FindNextFeatureSpeechOnsetAfter(Input.AudioFeatureFrames, LastCenter, RequiredTailGap, MaxTailSearch, TailGapStart, TailNextSpeechStart)
                    && TailGapStart > LastCenter + 0.045f)
                {
                    // Phrase-tail ownership: this event belongs to the current
                    // text phrase/sentence. If a real acoustic gap and next speech
                    // onset have been observed before the next text event, do not
                    // let the forced alignment path pull the current phrase
                    // tail into the next phrase.
                    TailGuardMaxCenter = TailGapStart - FMath::Min(EventSpan * 0.38f, 0.055f);
                    TailGuardMaxCenter = FMath::Max(LastCenter + 0.050f, TailGuardMaxCenter);
                    TailGuardReason = bNextSentenceBoundary
                        ? FName(TEXT("sentence_boundary_tail_guard"))
                        : FName(TEXT("phrase_boundary_tail_guard"));
                    if (Center > TailGuardMaxCenter)
                    {
                        Center = TailGuardMaxCenter;
                        PlacementReason = TailGuardReason;
                    }
                }
            }
        }

        // Streaming ownership remains with the online forced-alignment path.
        // Region pacing is visible here only as a bounded prior on phones that
        // already have acoustic evidence, plus an end-of-line final drain.
        const bool bUsedForcedAlignment = bHasPhoneEvidence;
        const bool bUsedObservedPlacement = bUsedForcedAlignment;
        const float Confidence = Alignment.PhoneMatchScores.IsValidIndex(PhoneIndex) ? Alignment.PhoneMatchScores[PhoneIndex] : 0.0f;
        const float PhoneStart = Alignment.PhoneStartSeconds.IsValidIndex(PhoneIndex) ? Alignment.PhoneStartSeconds[PhoneIndex] : 0.0f;
        const float PhoneEnd = Alignment.PhoneEndSeconds.IsValidIndex(PhoneIndex) ? Alignment.PhoneEndSeconds[PhoneIndex] : 0.0f;
        const float PhoneDuration = FMath::Max(PhoneEnd - PhoneStart, 0.0f);
        const bool bNextPhoneVisible = HasVisibleAlignedPhone(Alignment, PhoneIndex + 1);
        const bool bLaterPhonesVisible = Alignment.HighestAlignedPhoneIndex > PhoneIndex;
        const float LeadToPlayback = Center - Input.CurrentPlaybackSec;
        const float FirstAlignableObservedEnd = InOutTrack.RuntimeFirstAlignedObservedEndSeconds.IsValidIndex(PhoneIndex)
            ? InOutTrack.RuntimeFirstAlignedObservedEndSeconds[PhoneIndex]
            : -1.0f;
        const float AlignableLagSeconds = FirstAlignableObservedEnd >= 0.0f
            ? Input.ObservedAudioBufferEndSec - FirstAlignableObservedEnd
            : 0.0f;
        const float MinConfidence = CommitConfidenceThresholdForClass(PhoneClass, Pose);
        const bool bStableByConfidence = Confidence >= MinConfidence;
        const bool bStableByBoundary = bNextPhoneVisible || bLaterPhonesVisible;
        const bool bStableByDuration = PhoneDuration >= 0.040f || IsStrongPose(Pose);
        const bool bStableByLead = LeadToPlayback >= 0.040f;
        const bool bStableByLag = AlignableLagSeconds >= StreamingCommitLagSec;

        if (bUsedForcedAlignment && !Input.bInputStreamClosed && !Input.bPlaybackFinalized)
        {
            if ((!bStableByConfidence || !bStableByDuration) && !bStableByBoundary)
            {
                break;
            }
            if (!bStableByLead && !bStableByBoundary)
            {
                break;
            }
            if (!bStableByLag && !bStableByBoundary)
            {
                break;
            }
        }

        if (!bUsedObservedPlacement && !Input.bInputStreamClosed && !Input.bPlaybackFinalized)
        {
            // Streaming-safe: wait until the online Viterbi path exposes this phone.
            // Detector speech regions and word-pacing priors do not create live
            // commits on their own; they only shape already-observed phones.
            break;
        }

        if (!bUsedObservedPlacement && (Input.bInputStreamClosed || Input.bPlaybackFinalized))
        {
            // End-of-line drain: only after the stream is closed, use the
            // transcript duration model so weak/under-resolved audio cannot
            // permanently suppress planned visemes.
            PlacementReason = FName(TEXT("final_duration_drain"));
        }

        // Phrase-start guard: once a real acoustic gap/onset has been observed
        // at a text boundary, later placement stages may not pull the first event
        // of the new phrase/sentence backward into the previous phrase tail. This
        // runs after forced-alignment placement because the aligner can place
        // a boundary phone before the acoustic phrase onset.
        if (BoundaryGuardMinCenter >= 0.0f && Center < BoundaryGuardMinCenter - 0.010f)
        {
            Center = BoundaryGuardMinCenter;
            if (!BoundaryGuardReason.IsNone())
            {
                PlacementReason = BoundaryGuardReason;
            }
        }

        // Phrase-tail guard: later placement stages may not pull the final event
        // of a phrase/sentence across a detected pause into the following phrase.
        if (TailGuardMaxCenter >= 0.0f && Center > TailGuardMaxCenter + 0.010f)
        {
            Center = TailGuardMaxCenter;
            if (!TailGuardReason.IsNone())
            {
                PlacementReason = TailGuardReason;
            }
        }

        if (Center > CommitTimeHorizon + 0.001f)
        {
            break;
        }

        const float MinOrderedCenter = LastCenter + 0.050f;
        Center = FMath::Max(Center, MinOrderedCenter);

        if (!bUsedObservedPlacement && SameSentenceAsLast(InOutTrack, T))
        {
            const float MaxGap = MaxWallClockGapForPose(Pose);
            const float MaxSameSentenceCenter = LastCenter + MaxGap;
            if (Center > MaxSameSentenceCenter)
            {
                // This guard used to always pull later states forward.  That
                // avoided long frozen holds, but on slow/list delivery it also
                // compressed valid phones into the start of a spoken phrase,
                // creating the perceptual defect of "lost" animations while
                // speech continued.  Only apply the wall-clock cap when the gap
                // between events is acoustically a pause/non-speech region.  If
                // sustained speech evidence exists between the two centers, keep
                // the phone's aligned absolute time and let the performer bridge
                // the spoken phrase with real states.
                const bool bSpeechContinues = HasSustainedRuntimeSpeechBetween(
                    Input.AudioFeatureFrames,
                    LastCenter,
                    Center,
                    0.110f);
                const bool bLongSilence = HasLongRuntimeSilenceBetween(
                    Input.AudioFeatureFrames,
                    LastCenter,
                    Center,
                    0.140f);
                if (!bSpeechContinues || bLongSilence)
                {
                    Center = FMath::Max(MinOrderedCenter, MaxSameSentenceCenter);
                }
            }
        }

        if (Center < Input.CurrentPlaybackSec + 0.005f)
        {
            Center = FMath::Max(Input.CurrentPlaybackSec + 0.025f, LastCenter + 0.050f);
        }

        FOffgridAIAlignedVisemeEvent E;
        E.EventIndex = EventIndex;
        E.PoseID = Pose;
        E.Strength = StrengthForEvent(T, Pose);
        E.SourceWord = T.SourceText;
        E.WordIndex = T.WordIndex;
        E.PhraseIndex = T.PhraseIndex;
        E.bIsStrongVisibleEvent = T.bIsStrongVisibleEvent;
        E.TextCenterNorm = EventCenterNorm(T);
        E.TextDiagnosticCenterSeconds = E.TextCenterNorm * FMath::Max(Plan.EstimatedDurationSeconds, 0.001f);
        E.FinalRenderCenterSeconds = Center;
        E.RenderStartSeconds = FMath::Max(0.0f, Center - EventSpan * 0.5f);
        E.RenderEndSeconds = Center + EventSpan * 0.5f;
        E.SourcePhoneIndex = PhoneIndex;
        E.SourcePhoneBase = Plan.ExpectedPhones.IsValidIndex(PhoneIndex) ? Plan.ExpectedPhones[PhoneIndex].BasePhone : T.SourcePhoneBase;
        E.SourcePhoneClass = FName(*FOffgridAIOnlinePhoneAligner::PhoneClassToString(PhoneClass));
        E.AlignedPhoneStartSeconds = Alignment.PhoneStartSeconds.IsValidIndex(PhoneIndex)
            ? Alignment.PhoneStartSeconds[PhoneIndex]
            : 0.0f;
        E.AlignedPhoneEndSeconds = Alignment.PhoneEndSeconds.IsValidIndex(PhoneIndex)
            ? Alignment.PhoneEndSeconds[PhoneIndex]
            : 0.0f;
        E.AlignmentConfidence = Confidence;
        E.AlignmentScoreGap = Alignment.PhoneScoreGaps.IsValidIndex(PhoneIndex) ? Alignment.PhoneScoreGaps[PhoneIndex] : 0.0f;
        E.AlignmentObservedDurationSeconds = Alignment.PhoneObservedDurations.IsValidIndex(PhoneIndex) ? Alignment.PhoneObservedDurations[PhoneIndex] : 0.0f;
        E.AlignmentExpectedDurationSeconds = Alignment.PhoneExpectedDurations.IsValidIndex(PhoneIndex) ? Alignment.PhoneExpectedDurations[PhoneIndex] : 0.0f;
        E.AlignmentReason = bHasPhoneEvidence
            ? (Alignment.PhoneAdvanceReasons.IsValidIndex(PhoneIndex) && !Alignment.PhoneAdvanceReasons[PhoneIndex].IsNone()
                ? Alignment.PhoneAdvanceReasons[PhoneIndex]
                : FName(TEXT("streaming_forced_alignment")))
            : FName(TEXT("no_phone_evidence"));
        E.CommitPlaybackSeconds = Input.CurrentPlaybackSec;
        E.CommitLeadSeconds = Center - Input.CurrentPlaybackSec;
        E.CommitReason = PlacementReason;
        E.bCommitStableByConfidence = bStableByConfidence;
        E.bCommitStableByBoundary = bStableByBoundary;
        E.bCommitStableByDuration = bStableByDuration;
        E.bCommitStableByLead = bStableByLead;
        E.bCommitStableByLag = bStableByLag;
        E.CommitConfidenceThreshold = MinConfidence;
        E.CommitAlignableLagSeconds = AlignableLagSeconds;
        E.RequiredActiveElapsedSeconds = static_cast<float>(EventIndex + 1);
        E.ObservedActiveElapsedSeconds = bUsedForcedAlignment
            ? static_cast<float>(FMath::Max(Alignment.HighestAlignedPhoneIndex + 1, 0))
            : 0.0f;
        E.ActiveProgressDeficitSeconds = FMath::Max(0.0f, E.RequiredActiveElapsedSeconds - E.ObservedActiveElapsedSeconds);
        E.RequiredProgressNorm = E.RequiredActiveElapsedSeconds;
        E.ObservedProgressNorm = E.ObservedActiveElapsedSeconds;
        E.ActiveProgressRatio = E.RequiredActiveElapsedSeconds > 0.001f ? E.ObservedActiveElapsedSeconds / E.RequiredActiveElapsedSeconds : 1.0f;
        E.bMappedToObservedSpeech = bUsedObservedPlacement;
        E.SentenceIndex = T.SentenceIslandIndex;

        InOutTrack.Events.Add(E);
        LastCenter = Center;
    }

    if (InOutTrack.Events.Num() > 0)
    {
        float ObservedSpeechStart = 0.0f;
        float ObservedSpeechEnd = 0.0f;
        if (TryGetObservedSpeechEnvelope(Input.SpeechIslands, Input.ObservedAudioBufferEndSec, ObservedSpeechStart, ObservedSpeechEnd))
        {
            InOutTrack.SpeechStartSeconds = ObservedSpeechStart;
            InOutTrack.SpeechEndSeconds = ObservedSpeechEnd;
        }
        else
        {
            InOutTrack.SpeechStartSeconds = FirstSpeechStart >= 0.0f ? FirstSpeechStart : InOutTrack.Events[0].RenderStartSeconds;
            InOutTrack.SpeechEndSeconds = InOutTrack.Events.Last().RenderEndSeconds;
        }
    }
    bInOutTrackBuilt = InOutTrack.Events.Num() > 0 || Input.bInputStreamClosed || Input.bPlaybackFinalized;
}
