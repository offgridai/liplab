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
        return false;
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

static void BuildObservedRegionWordSpans(
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

    TArray<float> Weights;
    float TotalWeight = 0.0f;
    for (int32 WordIndex : Words)
    {
        const float Weight = WordExpectedWeight(Plan, WordIndex);
        Weights.Add(Weight);
        TotalWeight += Weight;
    }
    if (TotalWeight <= 0.001f)
    {
        return;
    }

    const float RegionDuration = FMath::Max(RegionEnd - RegionStart, 0.001f);
    TArray<float> Boundaries;
    Boundaries.SetNum(Words.Num() + 1);
    Boundaries[0] = RegionStart;
    Boundaries.Last() = RegionEnd;

    float Cumulative = 0.0f;
    for (int32 I = 1; I < Words.Num(); ++I)
    {
        Cumulative += Weights[I - 1];
        const float Prior = RegionStart + RegionDuration * (Cumulative / TotalWeight);

        const float SearchRadius = FMath::Clamp(RegionDuration * 0.060f, 0.070f, 0.160f);
        const float WindowStart = FMath::Max(RegionStart + 0.025f, Prior - SearchRadius);
        const float WindowEnd = FMath::Min(RegionEnd - 0.025f, Prior + SearchRadius);

        float ValleySec = Prior;
        bool bFoundValley = false;
        const float ValleyEvidence = BoundaryValleyEvidence(Frames, WindowStart, WindowEnd, ValleySec, bFoundValley);

        // Valleys are hints, not boundaries. They only attract the transcript
        // prior when local evidence is genuinely low. This budgeter uses the
        // observed region so far; it does not assume future sentence duration.
        float Attraction = 0.0f;
        if (bFoundValley)
        {
            Attraction = FMath::Clamp((0.145f - ValleyEvidence) / 0.145f, 0.0f, 0.42f);
        }
        Boundaries[I] = FMath::Lerp(Prior, ValleySec, Attraction);
    }

    // Preserve order and a small minimum word span. This is a monotonic time
    // allocation inside the currently observed speech region, not a valley detector.
    const float MinWordSpan = FMath::Clamp(RegionDuration / FMath::Max(static_cast<float>(Words.Num()) * 10.0f, 1.0f), 0.025f, 0.070f);
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
        if (Span.EndSec > Span.StartSec + 0.015f)
        {
            OutSpans.Add(Span.WordIndex, Span);
        }
    }
}

static bool TryGetObservedRegionEventCenter(
    const FOffgridAITextVisemePlan& Plan,
    const FOffgridAITextVisemeEvent& Event,
    const TMap<int32, FOffgridAIRegionWordSpan>& WordSpans,
    float& OutCenterSec,
    float& OutWordStartSec,
    float& OutWordEndSec)
{
    const FOffgridAIRegionWordSpan* Span = WordSpans.Find(Event.WordIndex);
    if (!Span)
    {
        return false;
    }

    const float Local = FMath::Clamp(Event.PhoneLocalNorm, 0.10f, 0.90f);
    OutWordStartSec = Span->StartSec;
    OutWordEndSec = Span->EndSec;
    OutCenterSec = FMath::Lerp(Span->StartSec, Span->EndSec, Local);
    return true;
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
    CommittedTrack.NPCID = NPCID;
    CommittedTrack.LineID = LineID;
    bBegun = true;
}

void FOffgridAILipsyncRuntimeSession::PushAudioPCM16(const TArray<uint8>& PCMChunk, int32 BytesToUse, int32 SampleRate, int32 NumChannels, int64 ChunkStartSample)
{
    if (!bBegun) return;
    Detector.AppendPCM16(PCMChunk, BytesToUse, SampleRate, NumChannels, ChunkStartSample);
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
}

void FOffgridAILipsyncRuntimeSession::Update(float CurrentPlaybackSec)
{
    PlaybackSec = FMath::Max(CurrentPlaybackSec, 0.0f);
    UpdatePlaybackGate(Detector.GetObservedAudioBufferEndSec());

    FOffgridAILipsyncRuntimeUpdateInput Input;
    Input.TextPlan = &TextPlan;
    Input.SpeechIslands = &Detector.GetIslands();
    Input.AudioFeatureFrames = &Detector.GetFeatureFrames();
    Input.CurrentPlaybackSec = PlaybackSec;
    Input.PrerollSec = PrerollSec;
    Input.ObservedAudioBufferEndSec = Detector.GetObservedAudioBufferEndSec();
    Input.bInputStreamClosed = bInputStreamClosed;
    Input.NPCID = NPCID;
    Input.LineID = LineID;
    FOffgridAILipsyncRuntimeAdapter::UpdateCommittedTrack(Input, CommittedTrack, bCommittedTrackBuilt);
    RecordAlignmentDiagnostics(PlaybackSec, false);
}

void FOffgridAILipsyncRuntimeSession::Finalize(float FinalPlaybackSec)
{
    PlaybackSec = FMath::Max(FinalPlaybackSec, PlaybackSec);
    bInputStreamClosed = true;
    Detector.Finalize(Detector.GetObservedAudioBufferEndSec());
    FOffgridAILipsyncRuntimeUpdateInput Input;
    Input.TextPlan = &TextPlan;
    Input.SpeechIslands = &Detector.GetIslands();
    Input.AudioFeatureFrames = &Detector.GetFeatureFrames();
    Input.CurrentPlaybackSec = PlaybackSec;
    Input.PrerollSec = PrerollSec;
    Input.ObservedAudioBufferEndSec = Detector.GetObservedAudioBufferEndSec();
    Input.bInputStreamClosed = true;
    Input.bPlaybackFinalized = true; // Final drain: represent every remaining planned viseme inside observed speech when possible.
    Input.NPCID = NPCID;
    Input.LineID = LineID;
    FOffgridAILipsyncRuntimeAdapter::UpdateCommittedTrack(Input, CommittedTrack, bCommittedTrackBuilt);
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

    // Observed-region pacing: once a detector speech island exists for a
    // transcript sentence, allocate the currently observed region across the
    // known word sequence using transcript duration priors. Local energy
    // valleys can nudge likely boundaries, but they cannot create, delete,
    // or reorder words.
    TMap<int32, FOffgridAIRegionWordSpan> ObservedRegionWordSpans;
    ObservedRegionWordSpans.Reset();
    {
        TArray<int32> SentenceIndices;
        for (const FOffgridAITextVisemeEvent& E : Plan.Events)
        {
            if (E.SentenceIslandIndex == INDEX_NONE)
            {
                continue;
            }

            bool bAlreadySeen = false;
            for (int32 ExistingSentenceIndex : SentenceIndices)
            {
                if (ExistingSentenceIndex == E.SentenceIslandIndex)
                {
                    bAlreadySeen = true;
                    break;
                }
            }
            if (!bAlreadySeen)
            {
                SentenceIndices.Add(E.SentenceIslandIndex);
            }
        }
        for (int32 SentenceIndex : SentenceIndices)
        {
            BuildObservedRegionWordSpans(
                Plan,
                Input.SpeechIslands,
                Input.AudioFeatureFrames,
                SentenceIndex,
                Input.ObservedAudioBufferEndSec,
                SentenceIndices.Num(),
                ObservedRegionWordSpans);
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
        float RegionWordStartSec = 0.0f;
        float RegionWordEndSec = 0.0f;
        float RegionPacedCenter = 0.0f;
        const bool bHasRegionPacing = TryGetObservedRegionEventCenter(
            Plan,
            T,
            ObservedRegionWordSpans,
            RegionPacedCenter,
            RegionWordStartSec,
            RegionWordEndSec);

        FName PlacementReason = FName(TEXT("duration_fallback"));
        bool bRegionPacingUsedForPlacement = false;

        if (bHasPhoneEvidence)
        {
            const float PhoneStart = Alignment.PhoneStartSeconds[PhoneIndex];
            const float PhoneEnd = Alignment.PhoneEndSeconds[PhoneIndex];
            const float PhoneCenter = CenterForAlignedPhone(Pose, PhoneClass, PhoneStart, PhoneEnd);
            Center = PhoneCenter;
            PlacementReason = FName(TEXT("streaming_forced_alignment"));

            // Region/word pacing is only a soft prior in the live streaming path.
            // It may gently regularize a phone that already has acoustic evidence,
            // but it must not create streaming commits by itself.  The previous
            // experiment made region pacing primary, which preserved speech onset
            // while destroying word/phone coverage because text allocation was
            // allowed to outrank the online aligner.
            if (bHasRegionPacing)
            {
                const float GuardStart = RegionWordStartSec - 0.160f;
                const float GuardEnd = RegionWordEndSec + 0.160f;
                if (PhoneCenter >= GuardStart && PhoneCenter <= GuardEnd)
                {
                    Center = FMath::Lerp(PhoneCenter, RegionPacedCenter, 0.18f);
                    PlacementReason = FName(TEXT("streaming_forced_alignment_region_prior"));
                    bRegionPacingUsedForPlacement = true;
                }
            }
        }
        else if (bHasRegionPacing && (Input.bInputStreamClosed || Input.bPlaybackFinalized))
        {
            // Final drain only: after no more audio can arrive, region pacing is a
            // deterministic recovery path for weak phones that never received
            // usable phone evidence.  It is deliberately disabled for live
            // streaming so offline batch generation cannot masquerade as the
            // realtime algorithm.
            Center = RegionPacedCenter;
            PlacementReason = FName(TEXT("final_region_word_drain"));
            bRegionPacingUsedForPlacement = true;
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
                const float RequiredGap = bSentenceBoundary ? 0.120f : (bSoftCommaBoundary ? 0.340f : 0.115f);
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
                    const float IslandRequiredGap = bSentenceBoundary ? 0.120f : (bSoftCommaBoundary ? 0.360f : 0.185f);
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
                const float RequiredTailGap = bNextSentenceBoundary ? 0.115f : (bNextSoftCommaBoundary ? 0.340f : 0.135f);
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
        const bool bUsedRegionPacing = bRegionPacingUsedForPlacement;
        const bool bUsedObservedPlacement = bUsedForcedAlignment || bUsedRegionPacing;
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
            : (bUsedRegionPacing ? RegionWordStartSec : 0.0f);
        E.AlignedPhoneEndSeconds = Alignment.PhoneEndSeconds.IsValidIndex(PhoneIndex)
            ? Alignment.PhoneEndSeconds[PhoneIndex]
            : (bUsedRegionPacing ? RegionWordEndSec : 0.0f);
        E.AlignmentConfidence = Confidence;
        E.AlignmentScoreGap = Alignment.PhoneScoreGaps.IsValidIndex(PhoneIndex) ? Alignment.PhoneScoreGaps[PhoneIndex] : 0.0f;
        E.AlignmentObservedDurationSeconds = Alignment.PhoneObservedDurations.IsValidIndex(PhoneIndex) ? Alignment.PhoneObservedDurations[PhoneIndex] : 0.0f;
        E.AlignmentExpectedDurationSeconds = Alignment.PhoneExpectedDurations.IsValidIndex(PhoneIndex) ? Alignment.PhoneExpectedDurations[PhoneIndex] : 0.0f;
        E.AlignmentReason = bHasPhoneEvidence
            ? (Alignment.PhoneAdvanceReasons.IsValidIndex(PhoneIndex) && !Alignment.PhoneAdvanceReasons[PhoneIndex].IsNone()
                ? Alignment.PhoneAdvanceReasons[PhoneIndex]
                : FName(TEXT("streaming_forced_alignment")))
            : (bUsedRegionPacing ? FName(TEXT("final_region_word_drain")) : FName(TEXT("no_phone_evidence")));
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
            : (bUsedRegionPacing ? static_cast<float>(EventIndex + 1) : 0.0f);
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
