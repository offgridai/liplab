#include "Lipsync/OffgridAILipsyncRuntimeAdapter.h"
#include "Lipsync/OffgridAIOnlinePhoneAligner.h"

namespace
{
static const FName OccupancyReason(TEXT("speech_occupancy_playhead"));
static const FName RegionClosedDropReason(TEXT("speech_region_closed_drop"));
static const FName MissingRegionDropReason(TEXT("speech_region_missing_drop"));
static constexpr float InterWordSpacerSeconds = 0.020f;
static constexpr float ActiveDurationScale = 0.90f;
static constexpr float LandmarkPacingMinRate = 0.88f;
static constexpr float LandmarkPacingMaxRate = 1.12f;
static constexpr float LandmarkPacingEmaAlpha = 0.35f;
static constexpr float LandmarkPacingMinSpacingSec = 0.040f;
static constexpr uint8 LandmarkMaskMbp = 1 << 0;
static constexpr uint8 LandmarkMaskFv = 1 << 1;
static constexpr uint8 LandmarkMaskW = 1 << 2;
static constexpr uint8 LandmarkMaskChJjSh = 1 << 3;
static constexpr uint8 LandmarkMaskCommaLull = 1 << 4;

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

static FString LandmarkTypeForPhoneBase(const FString& PhoneBase)
{
    if (PhoneBase == TEXT("M") || PhoneBase == TEXT("B") || PhoneBase == TEXT("P")) return TEXT("mbp");
    if (PhoneBase == TEXT("F") || PhoneBase == TEXT("V")) return TEXT("fv");
    if (PhoneBase == TEXT("W")) return TEXT("w");
    if (PhoneBase == TEXT("CH") || PhoneBase == TEXT("JH") || PhoneBase == TEXT("SH") || PhoneBase == TEXT("ZH")) return TEXT("chjjsh");
    return FString();
}

static uint8 LandmarkTypeMaskBit(const FString& Type)
{
    if (Type == TEXT("mbp")) return LandmarkMaskMbp;
    if (Type == TEXT("fv")) return LandmarkMaskFv;
    if (Type == TEXT("w")) return LandmarkMaskW;
    if (Type == TEXT("chjjsh")) return LandmarkMaskChJjSh;
    if (Type == TEXT("comma_lull")) return LandmarkMaskCommaLull;
    return 0;
}

static float LandmarkPacingThreshold(const FString& Type)
{
    if (Type == TEXT("mbp")) return 0.34f;
    if (Type == TEXT("fv")) return 0.34f;
    if (Type == TEXT("w")) return 0.30f;
    if (Type == TEXT("chjjsh")) return 0.38f;
    if (Type == TEXT("comma_lull")) return 0.44f;
    return 1.0f;
}

static float LandmarkPacingHalfWindowSec(const FString& Type)
{
    if (Type == TEXT("mbp")) return 0.12f;
    if (Type == TEXT("fv")) return 0.12f;
    if (Type == TEXT("w")) return 0.12f;
    if (Type == TEXT("chjjsh")) return 0.16f;
    if (Type == TEXT("comma_lull")) return 0.18f;
    return 0.10f;
}

static float LandmarkPacingReliability(const FString& Type)
{
    if (Type == TEXT("mbp")) return 0.70f;
    if (Type == TEXT("fv")) return 0.72f;
    if (Type == TEXT("w")) return 0.72f;
    if (Type == TEXT("chjjsh")) return 0.75f;
    if (Type == TEXT("comma_lull")) return 0.65f;
    return 0.60f;
}

static float Sat01(float X)
{
    return FMath::Clamp(X, 0.0f, 1.0f);
}

static float LandmarkFamilyScoreForType(const FString& Type, const FOffgridAIArticulatoryProbabilityField& Field)
{
    if (Type == TEXT("mbp"))
    {
        return FMath::Max(
            Field.PhoneScores.Bilabial,
            FMath::Max(Field.Closure * 0.70f, Field.Release * 0.35f));
    }
    if (Type == TEXT("fv"))
    {
        return FMath::Max(Field.PhoneScores.Labiodental, Field.Fricative * 0.85f);
    }
    if (Type == TEXT("w"))
    {
        return FMath::Max(
            FMath::Max(Field.PhoneScores.Glide, Field.PhoneScores.Liquid * 0.50f),
            FMath::Max(Field.PhoneScores.VowelRound * 0.85f, Field.Sonorant * 0.60f));
    }
    if (Type == TEXT("chjjsh"))
    {
        return FMath::Max(Field.PhoneScores.Sibilant, Field.PhoneScores.StopBurst * 0.55f);
    }
    return 0.0f;
}

static float CommaLullScoreAtFrame(const FOffgridAIStreamingAudioFeatureFrame& Frame)
{
    const FString PauseFamily = Frame.PauseFamily.ToString();
    const bool bPauseFamilySupport =
        !PauseFamily.IsEmpty()
        && PauseFamily != TEXT("none")
        && PauseFamily != TEXT("continuous_speech")
        && Frame.PauseFamilyConfidence >= 0.35f;
    const bool bLullGate =
        Frame.bLocalRMSValley
        || Frame.bStrongQuiet
        || Frame.SilenceAccumSec >= 0.030f
        || bPauseFamilySupport;
    if (!bLullGate)
    {
        return 0.0f;
    }

    const float LowRms = Sat01((0.12f - Frame.RMSNorm) / 0.12f);
    const float LowEvidence = Sat01((0.24f - Frame.SpeechEvidence) / 0.24f);
    const float SilenceAccum = Sat01(Frame.SilenceAccumSec / 0.080f);
    const float PauseConf = bPauseFamilySupport ? Frame.PauseFamilyConfidence : 0.0f;
    const float Valley = Frame.bLocalRMSValley ? 1.0f : 0.0f;
    const float StrongQuiet = Frame.bStrongQuiet ? 1.0f : 0.0f;
    return Sat01(
        LowRms * 0.28f +
        LowEvidence * 0.28f +
        SilenceAccum * 0.16f +
        PauseConf * 0.14f +
        Valley * 0.06f +
        StrongQuiet * 0.08f);
}

static float LandmarkTemplateScoreAtFrame(
    const FString& Type,
    const TArray<FOffgridAIStreamingAudioFeatureFrame>& Frames,
    int32 Index)
{
    if (!Frames.IsValidIndex(Index))
    {
        return 0.0f;
    }

    auto FrameFieldScore = [&](int32 SampleIndex) -> float {
        if (!Frames.IsValidIndex(SampleIndex))
        {
            return 0.0f;
        }
        return LandmarkFamilyScoreForType(
            Type,
            FOffgridAIOnlinePhoneAligner::BuildArticulatoryProbabilityField(Frames[SampleIndex]));
    };
    auto LocalMax = [&](int32 Lo, int32 Hi) -> float {
        float Best = 0.0f;
        for (int32 SampleIndex = Lo; SampleIndex <= Hi; ++SampleIndex)
        {
            Best = FMath::Max(Best, FrameFieldScore(SampleIndex));
        }
        return Best;
    };
    auto LocalMean = [&](int32 Lo, int32 Hi) -> float {
        float Sum = 0.0f;
        int32 Count = 0;
        for (int32 SampleIndex = Lo; SampleIndex <= Hi; ++SampleIndex)
        {
            if (!Frames.IsValidIndex(SampleIndex))
            {
                continue;
            }
            Sum += FrameFieldScore(SampleIndex);
            ++Count;
        }
        return Count > 0 ? (Sum / Count) : 0.0f;
    };

    const FOffgridAIStreamingAudioFeatureFrame& CenterFrame = Frames[Index];
    const FOffgridAIArticulatoryProbabilityField Field =
        FOffgridAIOnlinePhoneAligner::BuildArticulatoryProbabilityField(CenterFrame);
    const float CenterScore = FrameFieldScore(Index);

    if (Type == TEXT("mbp"))
    {
        const float ClosureBefore = LocalMax(Index - 2, Index);
        const float ReleaseAfter = FMath::Max(
            Field.Release,
            FMath::Max(FrameFieldScore(Index + 1), FrameFieldScore(Index + 2)));
        const float SpeechRise = Sat01((Frames.IsValidIndex(Index + 1) ? Frames[Index + 1].SpeechEvidence : 0.0f)
            - (Frames.IsValidIndex(Index - 1) ? Frames[Index - 1].SpeechEvidence : 0.0f)) * 1.6f;
        const float Burst = FMath::Max(
            Field.PhoneScores.StopBurst,
            Sat01((Frames.IsValidIndex(Index + 1) ? Frames[Index + 1].Flux : 0.0f) * 1.5f));
        return Sat01(
            CenterScore * 0.38f +
            ClosureBefore * 0.20f +
            ReleaseAfter * 0.20f +
            SpeechRise * 0.12f +
            Burst * 0.10f);
    }

    if (Type == TEXT("fv"))
    {
        const float Sustained = LocalMean(Index - 1, Index + 1);
        const float Friction = FMath::Max(Field.Fricative, LocalMax(Index - 1, Index + 1));
        const float HighNoise = FMath::Max(
            CenterFrame.HighBandNorm,
            Frames.IsValidIndex(Index + 1) ? Frames[Index + 1].HighBandNorm : 0.0f);
        return Sat01(
            CenterScore * 0.42f +
            Sustained * 0.26f +
            Friction * 0.20f +
            HighNoise * 0.12f);
    }

    if (Type == TEXT("w"))
    {
        const float Sustained = LocalMean(Index - 1, Index + 1);
        const float Sonorant = FMath::Max(Field.Sonorant, Field.PhoneScores.Glide);
        const float Smooth = Sat01((0.12f - CenterFrame.Flux) / 0.12f);
        const float Voicing = FMath::Max(
            CenterFrame.Periodicity,
            Frames.IsValidIndex(Index + 1) ? Frames[Index + 1].Periodicity : 0.0f);
        return Sat01(
            CenterScore * 0.38f +
            Sustained * 0.22f +
            Sonorant * 0.18f +
            Smooth * 0.10f +
            Voicing * 0.12f);
    }

    if (Type == TEXT("chjjsh"))
    {
        const float Sustained = LocalMean(Index - 1, Index + 1);
        const float Burst = FMath::Max(
            Field.PhoneScores.StopBurst,
            Sat01(CenterFrame.Flux * 1.4f));
        return Sat01(
            CenterScore * 0.46f +
            Sustained * 0.24f +
            Burst * 0.18f +
            CenterFrame.HighBandNorm * 0.12f);
    }

    if (Type == TEXT("comma_lull"))
    {
        return CommaLullScoreAtFrame(CenterFrame);
    }

    return CenterScore;
}

static float LandmarkTargetScoreAtFrame(
    const FString& Type,
    const FString& PhoneBase,
    const TArray<FOffgridAIStreamingAudioFeatureFrame>& Frames,
    int32 Index)
{
    if (!Frames.IsValidIndex(Index))
    {
        return 0.0f;
    }

    const float FamilyScore = LandmarkTemplateScoreAtFrame(Type, Frames, Index);
    if (FamilyScore <= 0.0f)
    {
        return 0.0f;
    }

    const FOffgridAIStreamingAudioFeatureFrame& Frame = Frames[Index];
    const FOffgridAIArticulatoryProbabilityField Field =
        FOffgridAIOnlinePhoneAligner::BuildArticulatoryProbabilityField(Frame);

    if (PhoneBase == TEXT("M"))
    {
        return Sat01(
            FamilyScore * 0.24f +
            Field.PhoneScores.Nasal * 0.28f +
            Field.Voiced * 0.16f +
            Field.Sonorant * 0.12f +
            Field.Closure * 0.10f +
            Sat01(1.0f - Field.PhoneScores.StopBurst) * 0.10f);
    }
    if (PhoneBase == TEXT("B"))
    {
        return Sat01(
            FamilyScore * 0.34f +
            Field.PhoneScores.Bilabial * 0.18f +
            Field.Closure * 0.16f +
            Field.Release * 0.12f +
            Field.Voiced * 0.12f +
            Field.PhoneScores.StopBurst * 0.08f);
    }
    if (PhoneBase == TEXT("P"))
    {
        return Sat01(
            FamilyScore * 0.34f +
            Field.PhoneScores.Bilabial * 0.18f +
            Field.Closure * 0.14f +
            Field.Release * 0.14f +
            Sat01(1.0f - Field.Voiced) * 0.12f +
            Field.PhoneScores.StopBurst * 0.08f);
    }
    if (PhoneBase == TEXT("F"))
    {
        return Sat01(
            FamilyScore * 0.34f +
            Field.PhoneScores.Labiodental * 0.18f +
            Field.Fricative * 0.18f +
            Frame.HighBandNorm * 0.10f +
            Sat01(1.0f - Field.Voiced) * 0.12f +
            Sat01(1.0f - Frame.LowBandNorm) * 0.08f);
    }
    if (PhoneBase == TEXT("V"))
    {
        return Sat01(
            FamilyScore * 0.34f +
            Field.PhoneScores.Labiodental * 0.18f +
            Field.Fricative * 0.16f +
            Field.Voiced * 0.16f +
            Frame.Periodicity * 0.10f +
            Frame.HighBandNorm * 0.06f);
    }
    if (PhoneBase == TEXT("W"))
    {
        return Sat01(
            FamilyScore * 0.54f +
            Field.PhoneScores.Glide * 0.14f +
            Field.PhoneScores.VowelRound * 0.10f +
            Field.Sonorant * 0.08f +
            Field.Voiced * 0.08f +
            Sat01(1.0f - Frame.Flux) * 0.06f);
    }
    if (PhoneBase == TEXT("CH"))
    {
        return Sat01(
            FamilyScore * 0.30f +
            Field.PhoneScores.Sibilant * 0.20f +
            Field.PhoneScores.StopBurst * 0.14f +
            Field.Release * 0.10f +
            Frame.HighBandNorm * 0.10f +
            Sat01(1.0f - Field.Voiced) * 0.16f);
    }
    if (PhoneBase == TEXT("JH"))
    {
        return Sat01(
            FamilyScore * 0.30f +
            Field.PhoneScores.Sibilant * 0.20f +
            Field.PhoneScores.StopBurst * 0.12f +
            Field.Release * 0.08f +
            Field.Voiced * 0.18f +
            Frame.Periodicity * 0.12f);
    }
    if (PhoneBase == TEXT("SH"))
    {
        return Sat01(
            FamilyScore * 0.30f +
            Field.PhoneScores.Sibilant * 0.22f +
            Field.Fricative * 0.12f +
            Frame.HighBandNorm * 0.14f +
            Sat01(1.0f - Field.Voiced) * 0.14f +
            Sat01(1.0f - Field.PhoneScores.StopBurst) * 0.08f);
    }
    if (PhoneBase == TEXT("ZH"))
    {
        return Sat01(
            FamilyScore * 0.30f +
            Field.PhoneScores.Sibilant * 0.20f +
            Field.Fricative * 0.12f +
            Frame.HighBandNorm * 0.12f +
            Field.Voiced * 0.16f +
            Frame.Periodicity * 0.10f);
    }

    return FamilyScore;
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
        return 0.160f;
    case EOffgridAIBoundaryPauseClass::HardBreakPause:
        return 0.260f;
    case EOffgridAIBoundaryPauseClass::None:
    default:
        return HoldSecondsForBoundary(C);
    }
}

static bool BoundaryRequiresObservedRegionTransition(TCHAR C, EOffgridAIBoundaryPauseClass PauseClass)
{
    if (PauseClass == EOffgridAIBoundaryPauseClass::HardBreakPause)
    {
        return true;
    }
    return C == TEXT('-');
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

    if (InOutState.bHoldActive)
    {
        InOutState.TotalPausedSec += DeltaSec;

        if (InOutState.bWaitingForSpeechResume)
        {
            if (EffectiveRegions.IsValidIndex(InOutState.ResumeRegionIndex)
                && PlaybackSec >= EffectiveRegions[InOutState.ResumeRegionIndex].StartSec)
            {
                InOutState.bHoldActive = false;
                InOutState.bWaitingForSpeechResume = false;
                InOutState.bResumeReanchorPending = InOutState.bRequireObservedRegionTransition;
                InOutState.ResumePlaybackSec = PlaybackSec;
                InOutState.bRequireObservedRegionTransition = false;
                InOutState.HoldRegionIndex = InOutState.ResumeRegionIndex;
                InOutState.ResumeRegionIndex = INDEX_NONE;
            }
        }
        else
        {
            if (EffectiveRegions.IsValidIndex(InOutState.HoldRegionIndex)
                && PlaybackSec >= EffectiveRegions[InOutState.HoldRegionIndex].EndSec)
            {
                if (InOutState.bRequireObservedRegionTransition)
                {
                    InOutState.bWaitingForSpeechResume = true;
                    InOutState.ResumeRegionIndex = InOutState.HoldRegionIndex + 1;
                }
                else
                {
                    InOutState.bHoldActive = false;
                    InOutState.bResumeReanchorPending = false;
                    InOutState.ResumeRegionIndex = INDEX_NONE;
                }
            }
            else if (!InOutState.bRequireObservedRegionTransition
                && PlaybackSec >= InOutState.HoldDeadlinePlaybackSec)
            {
                InOutState.bHoldActive = false;
                InOutState.bResumeReanchorPending = false;
                InOutState.ResumeRegionIndex = INDEX_NONE;
            }
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

    const float PlaybackOffsetSec = HoldState.PlaybackOriginSec + HoldState.TotalPausedSec + HoldState.PlaybackOffsetAdjustSec;
    for (int32 WordIndex = 0; WordIndex < WordStartActiveSeconds.Num(); ++WordIndex)
    {
        OutWordStartSeconds[WordIndex] = PlaybackOffsetSec + FMath::Max(WordStartActiveSeconds[WordIndex], 0.0f);
    }
}

static float ProjectActiveSecondsToClock(
    float ActiveSec,
    float PlaybackOffsetSec,
    const FOffgridAILandmarkPacingState& PacingState)
{
    if (!PacingState.bSeeded)
    {
        return PlaybackOffsetSec + FMath::Max(ActiveSec, 0.0f);
    }

    return PacingState.AnchorObservedSec
        + (FMath::Max(ActiveSec, 0.0f) - PacingState.AnchorPriorActiveSec) * PacingState.PlayRate;
}

static void BuildWordStartSecondsFromPacingProjection(
    const TArray<float>& WordStartActiveSeconds,
    float PlaybackOffsetSec,
    const FOffgridAILandmarkPacingState& PacingState,
    TArray<float>& OutWordStartSeconds)
{
    OutWordStartSeconds.Init(-1.0f, WordStartActiveSeconds.Num());
    for (int32 WordIndex = 0; WordIndex < WordStartActiveSeconds.Num(); ++WordIndex)
    {
        OutWordStartSeconds[WordIndex] =
            ProjectActiveSecondsToClock(WordStartActiveSeconds[WordIndex], PlaybackOffsetSec, PacingState);
    }
}

static int32 PlannedSpeechRegionCount(const FOffgridAITextVisemePlan& Plan)
{
    int32 RegionCount = 0;
    for (const int32 RegionIndex : Plan.WordSpeechRegionIndices)
    {
        RegionCount = FMath::Max(RegionCount, RegionIndex + 1);
    }
    for (const FOffgridAIExpectedPhone& Phone : Plan.ExpectedPhones)
    {
        RegionCount = FMath::Max(RegionCount, Phone.SpeechRegionIndex + 1);
    }
    return RegionCount;
}

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
    const int32 RegionCount = PlannedSpeechRegionCount(Plan);
    OutSpans.Init(FPlannedSpeechRegionActiveSpan(), RegionCount);
    for (int32 PhoneIndex = 0; PhoneIndex < Plan.ExpectedPhones.Num(); ++PhoneIndex)
    {
        const FOffgridAIExpectedPhone& Phone = Plan.ExpectedPhones[PhoneIndex];
        if (!OutSpans.IsValidIndex(Phone.SpeechRegionIndex)
            || !PhoneStartActiveSeconds.IsValidIndex(PhoneIndex)
            || !PhoneEndActiveSeconds.IsValidIndex(PhoneIndex))
        {
            continue;
        }

        FPlannedSpeechRegionActiveSpan& Span = OutSpans[Phone.SpeechRegionIndex];
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

static void ResetLandmarkPacingState(
    int32 SpeechRegionIndex,
    FOffgridAILandmarkPacingState& InOutState)
{
    InOutState = FOffgridAILandmarkPacingState();
    InOutState.SpeechRegionIndex = SpeechRegionIndex;
}

static void UpdateLandmarkPacingState(
    const FOffgridAILipsyncRuntimeUpdateInput& Input,
    const TArray<float>& PhoneCenterActiveSeconds,
    const TArray<float>& PhoneStartActiveSeconds,
    const TArray<float>& PhoneEndActiveSeconds,
    float PlaybackOffsetSec,
    int32 NextEventIndex,
    const TArray<FEffectiveSpeechRegion>& EffectiveRegions,
    const FOffgridAIPunctuationHoldState& HoldState,
    const FOffgridAITextVisemePlan& Plan,
    FOffgridAILandmarkPacingState& InOutState)
{
    if (!Input.bEnableLandmarkPacing
        || !Input.AudioFeatureFrames
        || Input.AudioFeatureFrames->Num() <= 0
        || !Plan.Events.IsValidIndex(NextEventIndex))
    {
        return;
    }

    if (HoldState.bHoldActive)
    {
        return;
    }

    const FOffgridAITextVisemeEvent& NextEvent = Plan.Events[NextEventIndex];
    const int32 ActiveRegionIndex = NextEvent.SpeechRegionIndex;
    if (InOutState.SpeechRegionIndex != ActiveRegionIndex)
    {
        ResetLandmarkPacingState(ActiveRegionIndex, InOutState);
    }

    const TArray<FOffgridAIStreamingAudioFeatureFrame>& Frames = *Input.AudioFeatureFrames;
    const float ObservedEndSec = FMath::Max(Input.ObservedAudioBufferEndSec, 0.0f);
    const float SearchStartSec = FMath::Max(Input.CurrentPlaybackSec - 0.030f, 0.0f);
    const float SearchEndSec = ObservedEndSec + 0.005f;
    const uint8 EnabledTypeMask = Input.LandmarkPacingTypeMask == 0 ? 0x1F : Input.LandmarkPacingTypeMask;

    float BestConfidence = 0.0f;
    int32 BestPhoneIndex = INDEX_NONE;
    int32 BestFrameIndex = INDEX_NONE;
    FString BestType;
    float BestObservedSec = 0.0f;
    float BestPriorActiveSec = 0.0f;

    const int32 StartPhoneIndex = FMath::Max(0, NextEvent.SourcePhoneGlobalIndex);
    const int32 MaxPhoneIndex = FMath::Min(Plan.ExpectedPhones.Num() - 1, StartPhoneIndex + 14);
    for (int32 PhoneIndex = StartPhoneIndex; PhoneIndex <= MaxPhoneIndex; ++PhoneIndex)
    {
        const FOffgridAIExpectedPhone& Phone = Plan.ExpectedPhones[PhoneIndex];
        if (Phone.SpeechRegionIndex != ActiveRegionIndex)
        {
            if (Phone.SpeechRegionIndex > ActiveRegionIndex)
            {
                break;
            }
            continue;
        }
        if (PhoneIndex <= InOutState.AnchorPhoneIndex || !PhoneCenterActiveSeconds.IsValidIndex(PhoneIndex))
        {
            continue;
        }

        const FString Type = LandmarkTypeForPhoneBase(Phone.BasePhone);
        if (Type.IsEmpty())
        {
            continue;
        }
        if ((EnabledTypeMask & LandmarkTypeMaskBit(Type)) == 0)
        {
            continue;
        }

        const float PredictedCenterSec = ProjectActiveSecondsToClock(
            PhoneCenterActiveSeconds[PhoneIndex],
            PlaybackOffsetSec,
            InOutState);
        const float HalfWindowSec = LandmarkPacingHalfWindowSec(Type);
        if (PredictedCenterSec + HalfWindowSec < SearchStartSec || PredictedCenterSec - HalfWindowSec > SearchEndSec)
        {
            continue;
        }

        float LocalBestConfidence = 0.0f;
        int32 LocalBestFrameIndex = INDEX_NONE;
        float LocalBestObservedSec = 0.0f;
        for (int32 FrameIndex = 0; FrameIndex < Frames.Num(); ++FrameIndex)
        {
            const FOffgridAIStreamingAudioFeatureFrame& Frame = Frames[FrameIndex];
            if (Frame.AudioBufferCenterSec < SearchStartSec || Frame.AudioBufferCenterSec > SearchEndSec)
            {
                continue;
            }
            const float DistanceSec = FMath::Abs(Frame.AudioBufferCenterSec - PredictedCenterSec);
            if (DistanceSec > HalfWindowSec)
            {
                continue;
            }

            const float Score = LandmarkTargetScoreAtFrame(Type, Phone.BasePhone, Frames, FrameIndex);
            const float Threshold = Input.bLandmarkPacingPermissive ? 0.0f : LandmarkPacingThreshold(Type);
            if (!Input.bLandmarkPacingPermissive && Score < Threshold)
            {
                continue;
            }

            const float DistancePenalty = Sat01(1.0f - DistanceSec / FMath::Max(HalfWindowSec, KINDA_SMALL_NUMBER));
            const float Confidence = Sat01(
                ((Score - Threshold) / FMath::Max(1.0f - Threshold, 0.001f)) * 0.82f
                + DistancePenalty * 0.18f) * (Input.bLandmarkPacingPermissive ? 1.0f : LandmarkPacingReliability(Type));
            if (Confidence > LocalBestConfidence)
            {
                LocalBestConfidence = Confidence;
                LocalBestFrameIndex = FrameIndex;
                LocalBestObservedSec = Frame.AudioBufferCenterSec;
            }
        }

        if (LocalBestConfidence > BestConfidence)
        {
            BestConfidence = LocalBestConfidence;
            BestPhoneIndex = PhoneIndex;
            BestFrameIndex = LocalBestFrameIndex;
            BestType = Type;
            BestObservedSec = LocalBestObservedSec;
            BestPriorActiveSec = PhoneCenterActiveSeconds[PhoneIndex];
        }
    }

    const int32 StartWordIndex = FMath::Max(0, NextEvent.WordIndex);
    const int32 MaxWordIndex = FMath::Min(Plan.WordBoundaryPunctuationAfter.Num() - 1, StartWordIndex + 6);
    if ((EnabledTypeMask & LandmarkMaskCommaLull) != 0)
    {
        for (int32 WordIndex = StartWordIndex; WordIndex <= MaxWordIndex; ++WordIndex)
        {
            if (!Plan.WordBoundaryPunctuationAfter.IsValidIndex(WordIndex)
                || Plan.WordBoundaryPunctuationAfter[WordIndex] != ',')
            {
                continue;
            }
            if (!Plan.WordPhoneEndIndices.IsValidIndex(WordIndex)
                || !Plan.WordPhoneBeginIndices.IsValidIndex(WordIndex + 1))
            {
                continue;
            }

            const int32 LeftPhoneEndIndex = Plan.WordPhoneEndIndices[WordIndex] - 1;
            const int32 RightPhoneIndex = Plan.WordPhoneBeginIndices[WordIndex + 1];
            if (LeftPhoneEndIndex < 0
                || RightPhoneIndex < 0
                || !Plan.ExpectedPhones.IsValidIndex(LeftPhoneEndIndex)
                || !Plan.ExpectedPhones.IsValidIndex(RightPhoneIndex))
            {
                continue;
            }
            if (Plan.ExpectedPhones[LeftPhoneEndIndex].SpeechRegionIndex != ActiveRegionIndex
                || Plan.ExpectedPhones[RightPhoneIndex].SpeechRegionIndex != ActiveRegionIndex)
            {
                continue;
            }
            if (RightPhoneIndex <= InOutState.AnchorPhoneIndex
                || !PhoneEndActiveSeconds.IsValidIndex(LeftPhoneEndIndex)
                || !PhoneStartActiveSeconds.IsValidIndex(RightPhoneIndex))
            {
                continue;
            }

            const FString Type = TEXT("comma_lull");
            const float PriorStartSec = PhoneEndActiveSeconds[LeftPhoneEndIndex];
            const float PriorEndSec = PhoneStartActiveSeconds[RightPhoneIndex];
            const float PriorCenterSec = 0.5f * (PriorStartSec + PriorEndSec);
            const float PredictedCenterSec = ProjectActiveSecondsToClock(
                PriorCenterSec,
                PlaybackOffsetSec,
                InOutState);
            const float HalfWindowSec = LandmarkPacingHalfWindowSec(Type);
            if (PredictedCenterSec + HalfWindowSec < SearchStartSec || PredictedCenterSec - HalfWindowSec > SearchEndSec)
            {
                continue;
            }

            float LocalBestConfidence = 0.0f;
            int32 LocalBestFrameIndex = INDEX_NONE;
            float LocalBestObservedSec = 0.0f;
            for (int32 FrameIndex = 0; FrameIndex < Frames.Num(); ++FrameIndex)
            {
                const FOffgridAIStreamingAudioFeatureFrame& Frame = Frames[FrameIndex];
                if (Frame.AudioBufferCenterSec < SearchStartSec || Frame.AudioBufferCenterSec > SearchEndSec)
                {
                    continue;
                }
                const float DistanceSec = FMath::Abs(Frame.AudioBufferCenterSec - PredictedCenterSec);
                if (DistanceSec > HalfWindowSec)
                {
                    continue;
                }

                const float Score = LandmarkTargetScoreAtFrame(Type, FString(), Frames, FrameIndex);
                const float Threshold = Input.bLandmarkPacingPermissive ? 0.0f : LandmarkPacingThreshold(Type);
                if (!Input.bLandmarkPacingPermissive && Score < Threshold)
                {
                    continue;
                }

                const float DistancePenalty = Sat01(1.0f - DistanceSec / FMath::Max(HalfWindowSec, KINDA_SMALL_NUMBER));
                const float Confidence = Sat01(
                    ((Score - Threshold) / FMath::Max(1.0f - Threshold, 0.001f)) * 0.82f
                    + DistancePenalty * 0.18f) * (Input.bLandmarkPacingPermissive ? 1.0f : LandmarkPacingReliability(Type));
                if (Confidence > LocalBestConfidence)
                {
                    LocalBestConfidence = Confidence;
                    LocalBestFrameIndex = FrameIndex;
                    LocalBestObservedSec = Frame.AudioBufferCenterSec;
                }
            }

            if (LocalBestConfidence > BestConfidence)
            {
                BestConfidence = LocalBestConfidence;
                BestPhoneIndex = RightPhoneIndex;
                BestFrameIndex = LocalBestFrameIndex;
                BestType = Type;
                BestObservedSec = LocalBestObservedSec;
                BestPriorActiveSec = PriorCenterSec;
            }
        }
    }

    const float MinAcceptConfidence = Input.bLandmarkPacingPermissive ? 0.0f : 0.55f;
    if (BestPhoneIndex == INDEX_NONE || BestFrameIndex == INDEX_NONE || BestConfidence < MinAcceptConfidence)
    {
        return;
    }

    if (InOutState.LastAppliedObservedSec >= 0.0f
        && BestObservedSec - InOutState.LastAppliedObservedSec < LandmarkPacingMinSpacingSec)
    {
        return;
    }

    if (!InOutState.bSeeded)
    {
        InOutState.bSeeded = true;
        InOutState.AnchorPhoneIndex = BestPhoneIndex;
        InOutState.AnchorPriorActiveSec = BestPriorActiveSec;
        InOutState.AnchorObservedSec = BestObservedSec;
        InOutState.PlayRate = 1.0f;
        InOutState.Confidence = BestConfidence;
        InOutState.LastAppliedObservedSec = BestObservedSec;
        InOutState.AnchorType = FName(*BestType);
        return;
    }

    const float PriorDeltaSec = BestPriorActiveSec - InOutState.AnchorPriorActiveSec;
    const float ObservedDeltaSec = BestObservedSec - InOutState.AnchorObservedSec;
    if (PriorDeltaSec < 0.050f || ObservedDeltaSec <= 0.0f)
    {
        return;
    }

    const float MeasuredRate = FMath::Clamp(
        ObservedDeltaSec / PriorDeltaSec,
        LandmarkPacingMinRate,
        LandmarkPacingMaxRate);
    const float Alpha = LandmarkPacingEmaAlpha * FMath::Clamp(BestConfidence, 0.35f, 1.0f);
    InOutState.PlayRate = FMath::Clamp(
        FMath::Lerp(InOutState.PlayRate, MeasuredRate, Alpha),
        LandmarkPacingMinRate,
        LandmarkPacingMaxRate);
    InOutState.AnchorPhoneIndex = BestPhoneIndex;
    InOutState.AnchorPriorActiveSec = BestPriorActiveSec;
    InOutState.AnchorObservedSec = BestObservedSec;
    InOutState.Confidence = BestConfidence;
    InOutState.LastAppliedObservedSec = BestObservedSec;
    InOutState.AnchorType = FName(*BestType);
}

static void BuildWordStartSecondsFromRegionPlayback(
    const FOffgridAITextVisemePlan& Plan,
    const TArray<float>& WordStartActiveSeconds,
    const TArray<FPlannedSpeechRegionActiveSpan>& PlannedRegionSpans,
    const TArray<FEffectiveSpeechRegion>& EffectiveRegions,
    TArray<float>& OutWordStartSeconds)
{
    OutWordStartSeconds.Init(-1.0f, Plan.WordSpeechRegionIndices.Num());
    for (int32 WordIndex = 0; WordIndex < Plan.WordSpeechRegionIndices.Num(); ++WordIndex)
    {
        const int32 RegionIndex = Plan.WordSpeechRegionIndices[WordIndex];
        if (!PlannedRegionSpans.IsValidIndex(RegionIndex) || !PlannedRegionSpans[RegionIndex].bValid)
        {
            continue;
        }

        const float RegionWordStartActiveSec = WordStartActiveSeconds.IsValidIndex(WordIndex)
            ? WordStartActiveSeconds[WordIndex]
            : PlannedRegionSpans[RegionIndex].StartActiveSec;
        const float LocalActiveSec = FMath::Max(
            RegionWordStartActiveSec - PlannedRegionSpans[RegionIndex].StartActiveSec,
            0.0f);
        if (EffectiveRegions.IsValidIndex(RegionIndex))
        {
            OutWordStartSeconds[WordIndex] = FMath::Clamp(
                EffectiveRegions[RegionIndex].StartSec + LocalActiveSec,
                EffectiveRegions[RegionIndex].StartSec,
                EffectiveRegions[RegionIndex].EndSec);
        }
    }
}

static void ClampWordStartSecondsToRegions(
    const FOffgridAITextVisemePlan& Plan,
    const TArray<FEffectiveSpeechRegion>& EffectiveRegions,
    TArray<float>& InOutWordStartSeconds)
{
    for (int32 WordIndex = 0; WordIndex < InOutWordStartSeconds.Num(); ++WordIndex)
    {
        if (InOutWordStartSeconds[WordIndex] < 0.0f || !Plan.WordSpeechRegionIndices.IsValidIndex(WordIndex))
        {
            continue;
        }
        const int32 RegionIndex = Plan.WordSpeechRegionIndices[WordIndex];
        if (!EffectiveRegions.IsValidIndex(RegionIndex))
        {
            continue;
        }

        InOutWordStartSeconds[WordIndex] = FMath::Clamp(
            InOutWordStartSeconds[WordIndex],
            EffectiveRegions[RegionIndex].StartSec,
            EffectiveRegions[RegionIndex].EndSec);
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
    bEnableLandmarkPacing = false;
    bLandmarkPacingPermissive = false;
    LandmarkPacingTypeMask = 0x1F;

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
    LandmarkPacingState = FOffgridAILandmarkPacingState();
}

void FOffgridAILipsyncRuntimeSession::BeginLine(const FOffgridAILipsyncRuntimeBeginInput& Input)
{
    Reset();
    NPCID = Input.NPCID;
    LineID = Input.LineID;
    DialogueText = Input.DialogueText;
    PrerollSec = FMath::Max(Input.PrerollSec, 0.0f);
    bEnableLandmarkPacing = Input.bEnableLandmarkPacing;
    bLandmarkPacingPermissive = Input.bLandmarkPacingPermissive;
    LandmarkPacingTypeMask = Input.LandmarkPacingTypeMask == 0 ? 0x1F : Input.LandmarkPacingTypeMask;
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
    Input.bEnableLandmarkPacing = bEnableLandmarkPacing;
    Input.bLandmarkPacingPermissive = bLandmarkPacingPermissive;
    Input.LandmarkPacingTypeMask = LandmarkPacingTypeMask;
    Input.NPCID = NPCID;
    Input.LineID = LineID;

    FOffgridAILipsyncRuntimeAdapter::UpdateCommittedTrack(Input, CommittedTrack, PunctuationHoldState, LandmarkPacingState, bCommittedTrackBuilt);
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
    Input.bEnableLandmarkPacing = bEnableLandmarkPacing;
    Input.bLandmarkPacingPermissive = bLandmarkPacingPermissive;
    Input.LandmarkPacingTypeMask = LandmarkPacingTypeMask;
    Input.NPCID = NPCID;
    Input.LineID = LineID;

    FOffgridAILipsyncRuntimeAdapter::UpdateCommittedTrack(Input, CommittedTrack, PunctuationHoldState, LandmarkPacingState, bCommittedTrackBuilt);
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

void FOffgridAILipsyncRuntimeAdapter::UpdateCommittedTrack(const FOffgridAILipsyncRuntimeUpdateInput& Input, FOffgridAIAlignedVisemeTrack& InOutTrack, FOffgridAIPunctuationHoldState& InOutHoldState, FOffgridAILandmarkPacingState& InOutLandmarkPacingState, bool& bInOutTrackBuilt)
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
    BuildEffectiveSpeechRegions(Input.SpeechRegions, Input.AudioFeatureFrames, ObservedEnd, bStreamSealed, EffectiveRegions);

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

    // Runtime scheduling:
    // 1. transcript owns viseme identity and order,
    // 2. speech onset owns when playback may start,
    // 3. punctuation opens a bounded hold window,
    // 4. a real speech break extends that hold until the next onset,
    // 5. otherwise playback resumes after the punctuation grace period.
    const float CommitLagSec = bStreamSealed ? 0.0f : 0.030f;
    const float MinLiveLeadSec = bPlaybackFinal ? 0.0f : 0.040f;
    const float MaxLiveLeadSec = bStreamSealed ? 999.0f : FMath::Max(Input.PrerollSec + 0.120f, 0.250f);

    int32 NextEventIndex = InOutTrack.Events.Num();
    float LastCenter = InOutTrack.Events.Num() > 0 ? InOutTrack.Events.Last().FinalRenderCenterSeconds : -1.0f;
    const float TotalPlannedActiveSec = FMath::Max(TotalPhoneActiveSeconds, 0.001f);

    if (InOutHoldState.bResumeReanchorPending && Plan.Events.IsValidIndex(NextEventIndex))
    {
        const FOffgridAITextVisemeEvent& ResumeEvent = Plan.Events[NextEventIndex];
        const int32 ResumePhoneIndex = ResumeEvent.SourcePhoneGlobalIndex;
        const float RequiredActiveSec = PhoneCenterActiveSeconds.IsValidIndex(ResumePhoneIndex)
            ? FMath::Max(PhoneCenterActiveSeconds[ResumePhoneIndex], 0.0f)
            : 0.0f;
        const float RequiredPhoneStartActiveSec = PhoneStartActiveSeconds.IsValidIndex(ResumePhoneIndex)
            ? FMath::Max(PhoneStartActiveSeconds[ResumePhoneIndex], 0.0f)
            : FMath::Max(RequiredActiveSec - 0.040f, 0.0f);
        const float RequiredPhoneEndActiveSec = PhoneEndActiveSeconds.IsValidIndex(ResumePhoneIndex)
            ? FMath::Max(PhoneEndActiveSeconds[ResumePhoneIndex], 0.0f)
            : (RequiredActiveSec + 0.040f);
        const float PlaybackOffsetSecBeforeAdjust =
            InOutHoldState.PlaybackOriginSec + InOutHoldState.TotalPausedSec + InOutHoldState.PlaybackOffsetAdjustSec;
        const float BaseStart = PlaybackOffsetSecBeforeAdjust + RequiredPhoneStartActiveSec;
        const float BaseEnd = PlaybackOffsetSecBeforeAdjust + RequiredPhoneEndActiveSec;
        const float BaseSpan = FMath::Max(BaseEnd - BaseStart, 0.020f);
        const float Span = FMath::Max(SpanForPose(ResumeEvent.PoseID), BaseSpan);
        const float Center = FMath::Max(
            PlaybackOffsetSecBeforeAdjust + RequiredActiveSec - LeadForPose(ResumeEvent.PoseID),
            0.0f);
        const float RenderStart = FMath::Max(Center - Span * 0.50f, 0.0f);
        const float DesiredRenderStart = FMath::Max(InOutHoldState.ResumePlaybackSec, 0.0f);
        if (RenderStart > DesiredRenderStart + 0.001f)
        {
            InOutHoldState.PlaybackOffsetAdjustSec += (DesiredRenderStart - RenderStart);
        }
        InOutHoldState.bResumeReanchorPending = false;
    }

    const float PlaybackOffsetSec =
        InOutHoldState.PlaybackOriginSec + InOutHoldState.TotalPausedSec + InOutHoldState.PlaybackOffsetAdjustSec;
    const float CommitSafeActiveSec = FMath::Max(InOutHoldState.ActivePlayheadSec - CommitLagSec, 0.0f);

    if (!Plan.Events.IsValidIndex(NextEventIndex)
        || Plan.Events[NextEventIndex].SpeechRegionIndex != InOutLandmarkPacingState.SpeechRegionIndex)
    {
        const int32 ResetRegionIndex = Plan.Events.IsValidIndex(NextEventIndex)
            ? Plan.Events[NextEventIndex].SpeechRegionIndex
            : INDEX_NONE;
        ResetLandmarkPacingState(ResetRegionIndex, InOutLandmarkPacingState);
    }

    if (InOutHoldState.bResumeReanchorPending || InOutHoldState.bHoldActive)
    {
        ResetLandmarkPacingState(InOutLandmarkPacingState.SpeechRegionIndex, InOutLandmarkPacingState);
    }

    UpdateLandmarkPacingState(
        Input,
        PhoneCenterActiveSeconds,
        PhoneStartActiveSeconds,
        PhoneEndActiveSeconds,
        PlaybackOffsetSec,
        NextEventIndex,
        EffectiveRegions,
        InOutHoldState,
        Plan,
        InOutLandmarkPacingState);

    if (Input.bEnableLandmarkPacing)
    {
        BuildWordStartSecondsFromPacingProjection(
            WordStartActiveSeconds,
            PlaybackOffsetSec,
            InOutLandmarkPacingState,
            WordStartSeconds);
    }
    else
    {
        BuildWordStartSecondsFromPlaybackClock(
            WordStartActiveSeconds,
            InOutHoldState,
            WordStartSeconds);
    }

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
                InOutHoldState.bWaitingForSpeechResume = false;
                InOutHoldState.bRequireObservedRegionTransition =
                    BoundaryRequiresObservedRegionTransition(Boundary, BoundaryPauseClass);
                InOutHoldState.BoundaryWordIndex = BoundaryWordIndex;
                InOutHoldState.HoldStartPlaybackSec = Input.CurrentPlaybackSec;
                InOutHoldState.HoldDeadlinePlaybackSec = Input.CurrentPlaybackSec + HoldSeconds;
                InOutHoldState.HoldRegionIndex = FMath::Max(FindRegionIndexAtPlayback(EffectiveRegions, Input.CurrentPlaybackSec), 0);
                InOutHoldState.ResumeRegionIndex = InOutHoldState.HoldRegionIndex + 1;
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

        float BaseStart = ProjectActiveSecondsToClock(RequiredPhoneStartActiveSec, PlaybackOffsetSec, InOutLandmarkPacingState);
        float Center = ProjectActiveSecondsToClock(RequiredActiveSec, PlaybackOffsetSec, InOutLandmarkPacingState);
        float BaseEnd = ProjectActiveSecondsToClock(RequiredPhoneEndActiveSec, PlaybackOffsetSec, InOutLandmarkPacingState);
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
        E.bLandmarkPacingEnabled = Input.bEnableLandmarkPacing;
        E.bLandmarkPacingSeeded = InOutLandmarkPacingState.bSeeded;
        E.LandmarkPacingRateAtCommit = InOutLandmarkPacingState.PlayRate;
        E.LandmarkPacingConfidenceAtCommit = InOutLandmarkPacingState.Confidence;
        E.LandmarkPacingAnchorObservedSeconds = InOutLandmarkPacingState.AnchorObservedSec;
        E.LandmarkPacingAnchorPriorActiveSeconds = InOutLandmarkPacingState.AnchorPriorActiveSec;
        E.LandmarkPacingAnchorType = InOutLandmarkPacingState.AnchorType;
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
