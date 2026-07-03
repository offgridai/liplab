#include "Lipsync/OffgridAIOnlinePhoneAligner.h"

#include <cfloat>

namespace
{
static float Sat(float X)
{
    return FMath::Clamp(X, 0.0f, 1.0f);
}

static float Smooth01(float X, float Edge0, float Edge1)
{
    if (Edge1 <= Edge0) return X >= Edge1 ? 1.0f : 0.0f;
    const float T = Sat((X - Edge0) / (Edge1 - Edge0));
    return T * T * (3.0f - 2.0f * T);
}

static float SpeechScore(const FOffgridAIStreamingAudioFeatureFrame& F)
{
    return Sat(FMath::Max(F.RMSNorm, FMath::Max(F.Flux * 0.55f, F.Periodicity * 0.45f)));
}

static bool IsSpeechFrame(const FOffgridAIStreamingAudioFeatureFrame& F)
{
    return SpeechScore(F) >= 0.055f;
}

static bool IsConfirmedByIsland(const TArray<FOffgridAIStreamingSpeechIsland>* Islands, const FOffgridAIStreamingAudioFeatureFrame& F)
{
    if (!Islands || Islands->Num() == 0)
    {
        return IsSpeechFrame(F);
    }

    const float Center = F.AudioBufferCenterSec;
    for (const FOffgridAIStreamingSpeechIsland& Island : *Islands)
    {
        const float IslandStart = Island.AudioBufferStartSec - 0.001f;
        const float IslandEnd = FMath::Max(Island.AudioBufferLastSpeechSec, Island.AudioBufferEndSec) + 0.001f;
        if (Center >= IslandStart && Center <= IslandEnd)
        {
            return true;
        }
    }
    return false;
}

static float MinDurationForClass(EOffgridAIPhoneClass C)
{
    switch (C)
    {
    case EOffgridAIPhoneClass::StopBurst: return 0.018f;
    case EOffgridAIPhoneClass::Bilabial: return 0.026f;
    case EOffgridAIPhoneClass::Labiodental: return 0.032f;
    case EOffgridAIPhoneClass::Sibilant: return 0.040f;
    case EOffgridAIPhoneClass::VowelOpen:
    case EOffgridAIPhoneClass::VowelFront:
    case EOffgridAIPhoneClass::VowelRound: return 0.048f;
    default: return 0.032f;
    }
}

static float MaxDurationForClass(EOffgridAIPhoneClass C)
{
    switch (C)
    {
    case EOffgridAIPhoneClass::StopBurst: return 0.090f;
    case EOffgridAIPhoneClass::Bilabial: return 0.120f;
    case EOffgridAIPhoneClass::Labiodental: return 0.135f;
    case EOffgridAIPhoneClass::Dental: return 0.120f;
    case EOffgridAIPhoneClass::Sibilant: return 0.165f;
    case EOffgridAIPhoneClass::Liquid: return 0.185f;
    case EOffgridAIPhoneClass::Glide: return 0.150f;
    case EOffgridAIPhoneClass::Nasal: return 0.165f;
    case EOffgridAIPhoneClass::VowelOpen: return 0.260f;
    case EOffgridAIPhoneClass::VowelFront: return 0.240f;
    case EOffgridAIPhoneClass::VowelRound: return 0.265f;
    default: return 0.155f;
    }
}

static float ExpectedDurationForPhone(const FOffgridAIExpectedPhone& Phone, EOffgridAIPhoneClass C)
{
    const float FromText = FMath::Clamp(Phone.WeightSeconds, 0.020f, 0.220f);
    const float ClassMid = (MinDurationForClass(C) + MaxDurationForClass(C)) * 0.5f;
    return FMath::Clamp(FromText * 0.65f + ClassMid * 0.35f, 0.020f, 0.240f);
}

struct FAlignFrame
{
    float Start = 0.0f;
    float End = 0.0f;
    float Center = 0.0f;
    float Duration = 0.0f;
    float ActiveStart = 0.0f;
    float ActiveEnd = 0.0f;
    float Speech = 0.0f;
    float BoundarySalience = 0.0f;
    FOffgridAIPhoneClassScores Scores;
};

struct FPhoneModel
{
    EOffgridAIPhoneClass Class = EOffgridAIPhoneClass::Unknown;
    float MinDuration = 0.025f;
    float ExpectedDuration = 0.075f;
    float MaxDuration = 0.180f;
    int32 MinFrames = 1;
    int32 ExpectedFrames = 2;
    int32 MaxFrames = 6;
};

struct FDPCell
{
    float Score = -FLT_MAX;
    int32 PrevFrame = INDEX_NONE;
};

static float SafeFrameSeconds(const TArray<FAlignFrame>& Frames)
{
    if (Frames.Num() == 0) return 0.020f;
    float Sum = 0.0f;
    for (const FAlignFrame& F : Frames)
    {
        Sum += FMath::Max(F.Duration, 0.001f);
    }
    return FMath::Clamp(Sum / FMath::Max(Frames.Num(), 1), 0.005f, 0.050f);
}

static float SegmentEmissionScore(const TArray<FAlignFrame>& Frames, int32 StartFrame, int32 EndFrame, EOffgridAIPhoneClass PhoneClass)
{
    if (!Frames.IsValidIndex(StartFrame) || !Frames.IsValidIndex(EndFrame) || EndFrame < StartFrame)
    {
        return -1000.0f;
    }

    float Sum = 0.0f;
    float Weight = 0.0f;
    for (int32 FrameIndex = StartFrame; FrameIndex <= EndFrame; ++FrameIndex)
    {
        const FAlignFrame& Frame = Frames[FrameIndex];
        const float ClassScore = FOffgridAIOnlinePhoneAligner::ScoreForClass(Frame.Scores, PhoneClass);
        const float UnknownScore = FOffgridAIOnlinePhoneAligner::ScoreForClass(Frame.Scores, EOffgridAIPhoneClass::Unknown);
        const float Speech = Frame.Speech;
        const float Combined = FMath::Max(ClassScore, UnknownScore * 0.15f) * 0.82f + Speech * 0.18f;
        const float W = FMath::Max(Frame.Duration, 0.001f);
        Sum += Combined * W;
        Weight += W;
    }
    return Weight > 0.0f ? Sum / Weight : 0.0f;
}

static float SegmentBestAlternateEmissionScore(const TArray<FAlignFrame>& Frames, int32 StartFrame, int32 EndFrame, EOffgridAIPhoneClass PhoneClass)
{
    float BestAlternate = -1000.0f;
    for (int32 ClassIndex = 0; ClassIndex <= static_cast<int32>(EOffgridAIPhoneClass::Unknown); ++ClassIndex)
    {
        const EOffgridAIPhoneClass CandidateClass = static_cast<EOffgridAIPhoneClass>(ClassIndex);
        if (CandidateClass == PhoneClass)
        {
            continue;
        }
        BestAlternate = FMath::Max(BestAlternate, SegmentEmissionScore(Frames, StartFrame, EndFrame, CandidateClass));
    }
    return BestAlternate;
}

static float SegmentActiveDurationSeconds(const TArray<FAlignFrame>& Frames, int32 StartFrame, int32 EndFrame)
{
    if (!Frames.IsValidIndex(StartFrame) || !Frames.IsValidIndex(EndFrame) || EndFrame < StartFrame)
    {
        return 0.0f;
    }
    return FMath::Max(Frames[EndFrame].ActiveEnd - Frames[StartFrame].ActiveStart, 0.0f);
}

static float LocalBoundarySalience(const TArray<FAlignFrame>& Frames, int32 BoundaryStartFrame)
{
    if (!Frames.IsValidIndex(BoundaryStartFrame))
    {
        return 0.0f;
    }

    float Best = 0.0f;
    const int32 First = FMath::Max(0, BoundaryStartFrame - 2);
    const int32 Last = FMath::Min(Frames.Num() - 1, BoundaryStartFrame + 2);
    for (int32 Index = First; Index <= Last; ++Index)
    {
        Best = FMath::Max(Best, Frames[Index].BoundarySalience);
    }
    return Sat(Best);
}

static float WordBoundaryTransitionReward(
    const TArray<FAlignFrame>& Frames,
    int32 BoundaryStartFrame,
    bool bCrossesExpectedWordBoundary)
{
    // Audio-only word-boundary salience is deliberately weak.  It is an independent
    // observation stream, not a replacement for the CMU phone path.  When the text
    // path crosses a word boundary, a nearby salience peak gets a small reward.  If
    // the crossing is placed in acoustically flat speech, apply only a tiny penalty.
    // Non-word phone transitions get a very small duplicate-peak penalty so strong
    // intra-word bursts do not attract every state boundary.
    const float S = LocalBoundarySalience(Frames, BoundaryStartFrame);
    if (bCrossesExpectedWordBoundary)
    {
        const float Reward = 0.44f * Smooth01(S, 0.20f, 0.78f);
        const float MissingPenalty = 0.07f * (1.0f - Smooth01(S, 0.08f, 0.28f));
        return Reward - MissingPenalty;
    }

    return -0.025f * Smooth01(S, 0.46f, 0.88f);
}

static float SegmentScore(const TArray<FAlignFrame>& Frames, int32 StartFrame, int32 EndFrame, const FPhoneModel& Phone)
{
    const float Emission = SegmentEmissionScore(Frames, StartFrame, EndFrame, Phone.Class);
    const float Alternate = SegmentBestAlternateEmissionScore(Frames, StartFrame, EndFrame, Phone.Class);
    const float Margin = Emission - Alternate;
    const float Dur = SegmentActiveDurationSeconds(Frames, StartFrame, EndFrame);
    const float Expected = FMath::Max(Phone.ExpectedDuration, 0.001f);
    const float DurErrorNorm = FMath::Abs(Dur - Expected) / Expected;
    const float TooShortPenalty = Dur < Phone.MinDuration ? (Phone.MinDuration - Dur) / FMath::Max(Phone.MinDuration, 0.001f) : 0.0f;
    const float TooLongPenalty = Dur > Phone.MaxDuration ? (Dur - Phone.MaxDuration) / FMath::Max(Phone.MaxDuration, 0.001f) : 0.0f;

    // Scores are negative costs in log-like units.  Emission dominates when the
    // local audio feature clearly supports the expected phone class; duration
    // priors prevent the path from dumping too many phones into one peak.
    return Emission * 2.75f
        + Margin * 1.15f
        - DurErrorNorm * 0.55f
        - TooShortPenalty * 1.05f
        - TooLongPenalty * 0.85f
        - 0.045f; // small state-advance cost
}

static float SourceBoundarySalienceAt(
    const TArray<FOffgridAIStreamingAudioFeatureFrame>& SourceFrames,
    const TArray<FOffgridAIStreamingSpeechIsland>* Islands,
    int32 Index)
{
    const int32 N = SourceFrames.Num();
    if (!SourceFrames.IsValidIndex(Index) || Index < 4 || Index + 4 >= N)
    {
        return 0.0f;
    }

    const FOffgridAIStreamingAudioFeatureFrame& F = SourceFrames[Index];
    if (!IsConfirmedByIsland(Islands, F))
    {
        return 0.0f;
    }

    auto Avg = [&](int32 Lo, int32 Hi, auto Getter) -> float
    {
        Lo = FMath::Max(0, Lo);
        Hi = FMath::Min(N - 1, Hi);
        if (Hi < Lo) return 0.0f;
        float Sum = 0.0f;
        int32 Count = 0;
        for (int32 I = Lo; I <= Hi; ++I)
        {
            Sum += Getter(SourceFrames[I]);
            ++Count;
        }
        return Count > 0 ? Sum / static_cast<float>(Count) : 0.0f;
    };

    auto MaxV = [&](int32 Lo, int32 Hi, auto Getter) -> float
    {
        Lo = FMath::Max(0, Lo);
        Hi = FMath::Min(N - 1, Hi);
        float Best = 0.0f;
        for (int32 I = Lo; I <= Hi; ++I)
        {
            Best = FMath::Max(Best, Getter(SourceFrames[I]));
        }
        return Best;
    };

    auto BandDistance = [](const FOffgridAIStreamingAudioFeatureFrame& A, const FOffgridAIStreamingAudioFeatureFrame& B) -> float
    {
        const float Dot = A.LowBandNorm * B.LowBandNorm + A.MidBandNorm * B.MidBandNorm + A.HighBandNorm * B.HighBandNorm + A.SpectralCentroidNorm * B.SpectralCentroidNorm;
        const float NA = FMath::Sqrt(A.LowBandNorm * A.LowBandNorm + A.MidBandNorm * A.MidBandNorm + A.HighBandNorm * A.HighBandNorm + A.SpectralCentroidNorm * A.SpectralCentroidNorm);
        const float NB = FMath::Sqrt(B.LowBandNorm * B.LowBandNorm + B.MidBandNorm * B.MidBandNorm + B.HighBandNorm * B.HighBandNorm + B.SpectralCentroidNorm * B.SpectralCentroidNorm);
        const float CosineDistance = (NA > 1e-6f && NB > 1e-6f) ? (1.0f - Dot / (NA * NB)) : 0.0f;
        const float L1 = FMath::Abs(A.LowBandNorm - B.LowBandNorm)
            + FMath::Abs(A.MidBandNorm - B.MidBandNorm)
            + FMath::Abs(A.HighBandNorm - B.HighBandNorm)
            + FMath::Abs(A.SpectralCentroidNorm - B.SpectralCentroidNorm);
        return Sat(0.58f * CosineDistance + 0.42f * L1);
    };

    const float PreRMS = Avg(Index - 4, Index - 1, [](const auto& X) { return X.RMSNorm; });
    const float PostRMS = Avg(Index + 1, Index + 4, [](const auto& X) { return X.RMSNorm; });
    const float PreSpeech = Avg(Index - 4, Index - 1, [](const auto& X) { return SpeechScore(X); });
    const float PostSpeech = Avg(Index + 1, Index + 4, [](const auto& X) { return SpeechScore(X); });
    const float PrePeriodic = Avg(Index - 4, Index - 1, [](const auto& X) { return X.Periodicity; });
    const float PostPeriodic = Avg(Index + 1, Index + 4, [](const auto& X) { return X.Periodicity; });
    const float LocalFlux = MaxV(Index - 2, Index + 2, [](const auto& X) { return X.Flux; });

    const float NeighbourEnergy = FMath::Max(PreRMS, PostRMS);
    const float EnergyValley = Sat((NeighbourEnergy - F.RMSNorm) / FMath::Max(NeighbourEnergy, 0.035f));
    const float LowEvidence = Sat(1.0f - SpeechScore(F) * 1.35f);
    const float SpectralNovelty = BandDistance(SourceFrames[Index - 3], SourceFrames[Index + 3]);
    const float VoicingChange = Sat(FMath::Abs(PostPeriodic - PrePeriodic) * 1.40f);
    const float Reengagement = Sat((PostSpeech - FMath::Min(PreSpeech, SpeechScore(F))) * 1.10f);
    const float FluxNovelty = Sat(LocalFlux * 0.75f + FMath::Max(0.0f, F.DeltaRMS) * 0.35f);

    const bool bMultiCue = ((EnergyValley > 0.22f) ? 1 : 0)
        + ((SpectralNovelty > 0.20f) ? 1 : 0)
        + ((VoicingChange > 0.22f) ? 1 : 0)
        + ((Reengagement > 0.18f) ? 1 : 0) >= 2;

    float Score = 0.0f;
    Score += EnergyValley * 0.26f;
    Score += SpectralNovelty * 0.30f;
    Score += VoicingChange * 0.18f;
    Score += Reengagement * 0.20f;
    Score += FluxNovelty * 0.10f;
    Score += LowEvidence * EnergyValley * 0.12f;
    Score += bMultiCue ? 0.08f : 0.0f;

    // Stable high-energy vowels create many spectral micro-events that are usually
    // intra-word.  Penalize them but do not zero them out; some fluent word joins
    // have no true energy valley.
    const float StableVoiced = Sat(F.Periodicity * 0.75f + F.RMSNorm * 0.45f - F.Flux * 0.30f);
    Score -= StableVoiced * (1.0f - EnergyValley) * 0.10f;
    return Sat(Score);
}

static void BuildPhoneModels(
    const FOffgridAITextVisemePlan& Plan,
    const TArray<FAlignFrame>& Frames,
    float SpeechRateScale,
    TArray<FPhoneModel>& OutModels)
{
    OutModels.Reset();
    const float FrameSec = SafeFrameSeconds(Frames);
    const int32 M = Frames.Num();
    const int32 N = Plan.ExpectedPhones.Num();
    const float SafeSpeechRateScale = FMath::Clamp(SpeechRateScale, 0.80f, 1.30f);
    const float ScaleAlpha = FMath::Clamp((SafeSpeechRateScale - 0.80f) / 0.50f, 0.0f, 1.0f);

    for (int32 PhoneIndex = 0; PhoneIndex < N; ++PhoneIndex)
    {
        const FOffgridAIExpectedPhone& Expected = Plan.ExpectedPhones[PhoneIndex];
        FPhoneModel Model;
        Model.Class = FOffgridAIOnlinePhoneAligner::ClassForPhoneBase(Expected.BasePhone);
        Model.MinDuration = MinDurationForClass(Model.Class) * FMath::Lerp(0.92f, 1.05f, ScaleAlpha);
        Model.ExpectedDuration = ExpectedDurationForPhone(Expected, Model.Class) * SafeSpeechRateScale;
        Model.MaxDuration = MaxDurationForClass(Model.Class) * FMath::Lerp(0.90f, 1.10f, ScaleAlpha);
        Model.MinDuration = FMath::Min(Model.MinDuration, Model.ExpectedDuration * 0.90f);
        Model.MaxDuration = FMath::Max(Model.MaxDuration, Model.ExpectedDuration * 1.10f);
        Model.MinFrames = 1;
        Model.ExpectedFrames = FMath::Max(1, FMath::RoundToInt(Model.ExpectedDuration / FrameSec));
        Model.MaxFrames = FMath::Max(Model.MinFrames, FMath::RoundToInt(Model.MaxDuration / FrameSec));

        // Keep enough remaining frames for the remaining phones.  This is a
        // streaming forced-alignment constraint, not an occupancy scheduler.
        const int32 RemainingPhones = N - PhoneIndex - 1;
        const int32 MaxAvailable = FMath::Max(1, M - RemainingPhones);
        Model.MaxFrames = FMath::Clamp(Model.MaxFrames, 1, MaxAvailable);
        OutModels.Add(Model);
    }
}

static void FillDurationFallback(const FOffgridAIOnlinePhoneAlignmentInput& Input, FOffgridAIOnlinePhoneAlignmentResult& Out)
{
    if (!Input.Plan || !Input.AudioFeatureFrames || Input.AudioFeatureFrames->Num() == 0)
    {
        return;
    }

    const FOffgridAITextVisemePlan& Plan = *Input.Plan;
    const float Start = Input.AudioFeatureFrames->Last().AudioBufferEndSec > 0.0f
        ? 0.0f
        : 0.0f;
    const float End = FMath::Max(Input.ObservedAudioEndSec, Plan.EstimatedDurationSeconds);
    if (End <= 0.001f)
    {
        return;
    }

    float TotalWeight = 0.0f;
    for (const FOffgridAIExpectedPhone& Phone : Plan.ExpectedPhones)
    {
        TotalWeight += FMath::Clamp(Phone.WeightSeconds, 0.025f, 0.180f);
    }
    if (TotalWeight <= 0.001f)
    {
        return;
    }

    float T = Start;
    for (int32 PhoneIndex = 0; PhoneIndex < Plan.ExpectedPhones.Num(); ++PhoneIndex)
    {
        const float Dur = (FMath::Clamp(Plan.ExpectedPhones[PhoneIndex].WeightSeconds, 0.025f, 0.180f) / TotalWeight) * End;
        const float S = T;
        const float E = (PhoneIndex == Plan.ExpectedPhones.Num() - 1) ? End : FMath::Min(End, T + Dur);
        Out.PhoneStartSeconds[PhoneIndex] = S;
        Out.PhoneEndSeconds[PhoneIndex] = E;
        Out.PhoneCenterSeconds[PhoneIndex] = (S + E) * 0.5f;
        Out.PhoneMatchScores[PhoneIndex] = 0.05f;
        Out.PhoneAdvanceReasons[PhoneIndex] = FName(TEXT("final_duration_drain"));
        Out.HighestAlignedPhoneIndex = PhoneIndex;
        Out.HighestAlignedWordIndex = Plan.ExpectedPhones[PhoneIndex].WordIndex;
        T = E;
    }
    Out.bHasSpeechEvidence = true;
    Out.AdvanceReason = FName(TEXT("final_duration_drain"));
}
}

EOffgridAIPhoneClass FOffgridAIOnlinePhoneAligner::ClassForPhoneBase(const FString& PhoneBase)
{
    const FString P = PhoneBase.ToUpper();
    if (P.IsEmpty()) return EOffgridAIPhoneClass::Unknown;

    if (P == TEXT("AA") || P == TEXT("AE") || P == TEXT("AH") || P == TEXT("AW") || P == TEXT("AY")) return EOffgridAIPhoneClass::VowelOpen;
    if (P == TEXT("EH") || P == TEXT("ER") || P == TEXT("EY") || P == TEXT("IH") || P == TEXT("IY")) return EOffgridAIPhoneClass::VowelFront;
    if (P == TEXT("AO") || P == TEXT("OW") || P == TEXT("OY") || P == TEXT("UH") || P == TEXT("UW")) return EOffgridAIPhoneClass::VowelRound;

    if (P == TEXT("B") || P == TEXT("M") || P == TEXT("P")) return EOffgridAIPhoneClass::Bilabial;
    if (P == TEXT("F") || P == TEXT("V")) return EOffgridAIPhoneClass::Labiodental;
    if (P == TEXT("TH") || P == TEXT("DH")) return EOffgridAIPhoneClass::Dental;
    if (P == TEXT("S") || P == TEXT("Z") || P == TEXT("SH") || P == TEXT("ZH") || P == TEXT("CH") || P == TEXT("JH")) return EOffgridAIPhoneClass::Sibilant;
    if (P == TEXT("T") || P == TEXT("D") || P == TEXT("K") || P == TEXT("G") || P == TEXT("HH")) return EOffgridAIPhoneClass::StopBurst;
    if (P == TEXT("L") || P == TEXT("R")) return EOffgridAIPhoneClass::Liquid;
    if (P == TEXT("W") || P == TEXT("Y")) return EOffgridAIPhoneClass::Glide;
    if (P == TEXT("N") || P == TEXT("NG")) return EOffgridAIPhoneClass::Nasal;

    return EOffgridAIPhoneClass::Unknown;
}

FOffgridAIArticulatoryProbabilityField FOffgridAIOnlinePhoneAligner::BuildArticulatoryProbabilityField(const FOffgridAIStreamingAudioFeatureFrame& F)
{
    FOffgridAIArticulatoryProbabilityField A;

    const float Speech = SpeechScore(F);
    const float Low = Sat(F.LowBandNorm);
    const float Mid = Sat(F.MidBandNorm);
    const float High = Sat(F.HighBandNorm);
    const float Centroid = Sat(F.SpectralCentroidNorm);
    const float LowCentroid = 1.0f - Centroid;
    const float HighCentroid = Centroid;
    const float LowHighTilt = Sat(Low - High + 0.42f);
    const float HighLowTilt = Sat(High - Low + 0.42f);

    const float PositiveEnergySlope = Sat(FMath::Max(F.DeltaRMS, 0.0f) * 1.65f);
    const float NegativeEnergySlope = Sat(FMath::Max(-F.DeltaRMS, 0.0f) * 1.65f);
    const float FluxPeak = Sat(F.Flux * 1.65f + (F.bLocalFluxPeak ? 0.28f : 0.0f));
    const float Valley = Sat((F.bLocalRMSValley ? 0.42f : 0.0f) + Smooth01(0.22f - F.RMSNorm, 0.0f, 0.22f) * 0.44f);

    A.Speech = Speech;
    A.Silence = Sat((1.0f - Speech) * 0.86f + (F.RMSNorm < 0.025f ? 0.25f : 0.0f));
    A.Voiced = Sat(F.Periodicity * 1.20f + F.RMSNorm * 0.30f - F.Flux * 0.16f - High * 0.08f);
    A.Fricative = Sat(((1.0f - F.Periodicity) * 0.48f + F.ZCR * 0.36f + High * 0.28f) * 0.64f + HighCentroid * 0.28f + F.Flux * 0.18f);
    A.Closure = Sat(Valley * 0.68f + NegativeEnergySlope * 0.18f + LowHighTilt * 0.16f - High * 0.10f);
    A.Release = Sat(FluxPeak * 0.70f + PositiveEnergySlope * 0.30f + High * 0.10f);
    A.Vowel = Sat(A.Voiced * 0.68f + F.RMSNorm * 0.36f - F.Flux * 0.24f - F.ZCR * 0.12f);
    A.Sonorant = Sat(A.Voiced * 0.56f + Low * 0.18f + Mid * 0.18f - A.Fricative * 0.12f - A.Closure * 0.08f);
    A.LowTilt = LowHighTilt;
    A.HighTilt = HighLowTilt;
    A.SpectralChange = Sat(F.Flux * 0.44f + F.ZCR * 0.22f + FMath::Abs(High - Low) * 0.16f + FMath::Abs(Centroid - 0.5f) * 0.10f);
    A.EnergyChange = Sat(PositiveEnergySlope * 0.55f + NegativeEnergySlope * 0.45f + (F.bLocalRMSValley ? 0.12f : 0.0f));

    // Transition probability is deliberately generic: it means "the articulatory
    // state is changing here", not "this exact frame is a word boundary".  The
    // runtime adapter aggregates this field over a window and combines it with
    // CMU-derived priors before it becomes a lexical-transition hint.
    A.Transition = Sat(0.30f * A.Closure + 0.26f * A.Release + 0.20f * A.SpectralChange + 0.14f * A.EnergyChange + 0.10f * FMath::Abs(A.Voiced - A.Fricative));

    const float SustainedVowel = Sat((F.RMSNorm - 0.18f) / 0.34f) * Sat((A.Vowel - 0.48f) / 0.36f) * Sat((0.38f - A.Transition) / 0.38f);
    const float FluxOnly = Sat(A.Release - 0.48f) * Sat((0.34f - A.Closure) / 0.34f) * Sat((0.34f - A.SpectralChange) / 0.34f);
    const float Flat = Sat((0.12f - A.EnergyChange) / 0.12f) * Sat((0.12f - A.SpectralChange) / 0.12f) * Sat((0.14f - A.Release) / 0.14f);
    A.RedHerring = Sat(0.48f * SustainedVowel + 0.34f * FluxOnly + 0.18f * Flat);

    FOffgridAIPhoneClassScores& S = A.PhoneScores;
    S.Silence = A.Silence;
    S.VowelOpen = Sat(A.Vowel * (0.48f + Low * 0.28f + Mid * 0.30f + F.RMSNorm * 0.12f - High * 0.08f));
    S.VowelFront = Sat(A.Vowel * (0.42f + Mid * 0.34f + Centroid * 0.26f + High * 0.10f - LowHighTilt * 0.08f));
    S.VowelRound = Sat(A.Vowel * (0.46f + LowCentroid * 0.32f + Low * 0.28f + LowHighTilt * 0.16f - High * 0.10f));
    S.Bilabial = Sat(A.Closure * 0.66f + A.Release * 0.18f + LowHighTilt * 0.16f + LowCentroid * 0.10f - A.Fricative * 0.18f);
    S.Labiodental = Sat(A.Fricative * 0.48f + Mid * 0.24f + LowHighTilt * 0.18f + LowCentroid * 0.10f - High * 0.10f);
    S.Dental = Sat(A.Fricative * 0.42f + Mid * 0.20f + F.ZCR * 0.22f + A.Release * 0.10f);
    S.Sibilant = Sat(A.Fricative * 0.78f + High * 0.34f + HighCentroid * 0.26f - Low * 0.10f);
    S.StopBurst = Sat(A.Release * 0.78f + Mid * 0.16f + High * 0.14f + A.Closure * 0.08f);
    S.Liquid = Sat(A.Voiced * 0.50f + Mid * 0.24f + Centroid * 0.14f + F.RMSNorm * 0.10f - A.Closure * 0.10f);
    S.Glide = Sat(A.Voiced * 0.48f + LowCentroid * 0.18f + Mid * 0.18f + F.RMSNorm * 0.08f - A.Release * 0.10f);
    S.Nasal = Sat(A.Voiced * 0.54f + Low * 0.34f + LowCentroid * 0.22f - High * 0.18f - A.Fricative * 0.08f);
    S.Unknown = Sat(Speech * 0.22f + 0.035f);
    return A;
}

FOffgridAIPhoneClassScores FOffgridAIOnlinePhoneAligner::ScoreFramePhoneClasses(const FOffgridAIStreamingAudioFeatureFrame& F)
{
    return BuildArticulatoryProbabilityField(F).PhoneScores;
}

float FOffgridAIOnlinePhoneAligner::ScoreForClass(const FOffgridAIPhoneClassScores& Scores, EOffgridAIPhoneClass PhoneClass)
{
    switch (PhoneClass)
    {
    case EOffgridAIPhoneClass::Silence: return Scores.Silence;
    case EOffgridAIPhoneClass::VowelOpen: return Scores.VowelOpen;
    case EOffgridAIPhoneClass::VowelFront: return Scores.VowelFront;
    case EOffgridAIPhoneClass::VowelRound: return Scores.VowelRound;
    case EOffgridAIPhoneClass::Bilabial: return Scores.Bilabial;
    case EOffgridAIPhoneClass::Labiodental: return Scores.Labiodental;
    case EOffgridAIPhoneClass::Dental: return Scores.Dental;
    case EOffgridAIPhoneClass::Sibilant: return Scores.Sibilant;
    case EOffgridAIPhoneClass::StopBurst: return Scores.StopBurst;
    case EOffgridAIPhoneClass::Liquid: return Scores.Liquid;
    case EOffgridAIPhoneClass::Glide: return Scores.Glide;
    case EOffgridAIPhoneClass::Nasal: return Scores.Nasal;
    default: return Scores.Unknown;
    }
}

FString FOffgridAIOnlinePhoneAligner::PhoneClassToString(EOffgridAIPhoneClass PhoneClass)
{
    switch (PhoneClass)
    {
    case EOffgridAIPhoneClass::Silence: return TEXT("silence");
    case EOffgridAIPhoneClass::VowelOpen: return TEXT("vowel_open");
    case EOffgridAIPhoneClass::VowelFront: return TEXT("vowel_front");
    case EOffgridAIPhoneClass::VowelRound: return TEXT("vowel_round");
    case EOffgridAIPhoneClass::Bilabial: return TEXT("bilabial");
    case EOffgridAIPhoneClass::Labiodental: return TEXT("labiodental");
    case EOffgridAIPhoneClass::Dental: return TEXT("dental");
    case EOffgridAIPhoneClass::Sibilant: return TEXT("sibilant");
    case EOffgridAIPhoneClass::StopBurst: return TEXT("stop_burst");
    case EOffgridAIPhoneClass::Liquid: return TEXT("liquid");
    case EOffgridAIPhoneClass::Glide: return TEXT("glide");
    case EOffgridAIPhoneClass::Nasal: return TEXT("nasal");
    default: return TEXT("unknown");
    }
}

int32 FOffgridAIOnlinePhoneAligner::FindPhoneForEvent(const FOffgridAITextVisemePlan& Plan, const FOffgridAITextVisemeEvent& Event)
{
    if (Plan.ExpectedPhones.IsValidIndex(Event.SourcePhoneGlobalIndex))
    {
        const FOffgridAIExpectedPhone& Expected = Plan.ExpectedPhones[Event.SourcePhoneGlobalIndex];
        if (Expected.WordIndex == Event.WordIndex)
        {
            return Expected.PhoneIndex;
        }
    }

    int32 Fallback = INDEX_NONE;
    int32 BaseMatch = INDEX_NONE;
    int32 ClassMatch = INDEX_NONE;
    const EOffgridAIPhoneClass EventClass = ClassForPhoneBase(Event.SourcePhoneBase);
    for (const FOffgridAIExpectedPhone& P : Plan.ExpectedPhones)
    {
        if (P.WordIndex == Event.WordIndex)
        {
            if (Fallback == INDEX_NONE) Fallback = P.PhoneIndex;
            if (Event.SourcePhoneIndex != INDEX_NONE && P.WordPhoneIndex == Event.SourcePhoneIndex)
            {
                return P.PhoneIndex;
            }
            if (BaseMatch == INDEX_NONE
                && !Event.SourcePhoneBase.IsEmpty()
                && P.BasePhone.ToUpper() == Event.SourcePhoneBase.ToUpper())
            {
                BaseMatch = P.PhoneIndex;
            }
            if (ClassMatch == INDEX_NONE && EventClass != EOffgridAIPhoneClass::Unknown && ClassForPhoneBase(P.BasePhone) == EventClass)
            {
                ClassMatch = P.PhoneIndex;
            }
        }
    }
    if (BaseMatch != INDEX_NONE) return BaseMatch;
    if (ClassMatch != INDEX_NONE) return ClassMatch;
    return Fallback;
}

FOffgridAIOnlinePhoneAlignmentResult FOffgridAIOnlinePhoneAligner::Compute(const FOffgridAIOnlinePhoneAlignmentInput& Input)
{
    FOffgridAIOnlinePhoneAlignmentResult Out;
    if (!Input.Plan || Input.Plan->ExpectedPhones.Num() == 0)
    {
        return Out;
    }

    const FOffgridAITextVisemePlan& Plan = *Input.Plan;
    const int32 PhoneCount = Plan.ExpectedPhones.Num();
    Out.PhoneStartSeconds.Init(-1.0f, PhoneCount);
    Out.PhoneCenterSeconds.Init(-1.0f, PhoneCount);
    Out.PhoneEndSeconds.Init(-1.0f, PhoneCount);
    Out.PhoneMatchScores.Init(0.0f, PhoneCount);
    Out.PhoneScoreGaps.Init(0.0f, PhoneCount);
    Out.PhoneObservedDurations.Init(0.0f, PhoneCount);
    Out.PhoneExpectedDurations.Init(0.0f, PhoneCount);
    Out.PhoneWordBoundarySalience.Init(0.0f, PhoneCount);
    Out.PhoneAdvanceReasons.Init(NAME_None, PhoneCount);

    if (!Input.AudioFeatureFrames || Input.AudioFeatureFrames->Num() == 0)
    {
        return Out;
    }

    const float VisibleEnd = Input.bFinal
        ? Input.ObservedAudioEndSec + 0.001f
        : FMath::Min(Input.ObservedAudioEndSec, Input.PlaybackSec + FMath::Max(Input.LookaheadSec, 0.0f)) + 0.001f;
    const float StableEnd = Input.bFinal
        ? VisibleEnd
        : FMath::Max(0.0f, VisibleEnd - FMath::Clamp(Input.CommitLagSec, 0.020f, 0.250f));

    TArray<float> SourceBoundarySalience;
    SourceBoundarySalience.SetNum(Input.AudioFeatureFrames->Num());
    for (int32 SourceIndex = 0; SourceIndex < Input.AudioFeatureFrames->Num(); ++SourceIndex)
    {
        SourceBoundarySalience[SourceIndex] = SourceBoundarySalienceAt(*Input.AudioFeatureFrames, Input.SpeechIslands, SourceIndex);
    }

    TArray<FAlignFrame> Frames;
    float ActiveCursor = 0.0f;
    for (int32 SourceIndex = 0; SourceIndex < Input.AudioFeatureFrames->Num(); ++SourceIndex)
    {
        const FOffgridAIStreamingAudioFeatureFrame& SourceFrame = (*Input.AudioFeatureFrames)[SourceIndex];
        if (SourceFrame.AudioBufferEndSec > VisibleEnd)
        {
            break;
        }
        if (!IsSpeechFrame(SourceFrame) || !IsConfirmedByIsland(Input.SpeechIslands, SourceFrame))
        {
            continue;
        }

        FAlignFrame Frame;
        Frame.Start = SourceFrame.AudioBufferStartSec;
        Frame.End = SourceFrame.AudioBufferEndSec;
        Frame.Center = SourceFrame.AudioBufferCenterSec;
        Frame.Duration = FMath::Max(SourceFrame.AudioBufferEndSec - SourceFrame.AudioBufferStartSec, 0.001f);
        Frame.ActiveStart = ActiveCursor;
        Frame.ActiveEnd = ActiveCursor + Frame.Duration;
        Frame.Speech = SpeechScore(SourceFrame);
        Frame.BoundarySalience = SourceBoundarySalience.IsValidIndex(SourceIndex) ? SourceBoundarySalience[SourceIndex] : 0.0f;
        Frame.Scores = ScoreFramePhoneClasses(SourceFrame);
        Frames.Add(Frame);
        ActiveCursor = Frame.ActiveEnd;
    }

    if (Frames.Num() == 0)
    {
        if (Input.bFinal)
        {
            FillDurationFallback(Input, Out);
        }
        return Out;
    }

    float ObservedSpeechSeconds = 0.0f;
    for (const FAlignFrame& Frame : Frames)
    {
        ObservedSpeechSeconds += FMath::Max(Frame.Duration, 0.0f);
    }
    Out.bHasSpeechEvidence = true;
    Out.VisibleSpeechSeconds = ObservedSpeechSeconds;

    const int32 M = Frames.Num();
    if (M < 1 || PhoneCount < 1)
    {
        return Out;
    }

    int32 PhoneLimit = PhoneCount;
    if (!Input.bFinal)
    {
        // Streaming forced alignment must not compress the whole transcript into
        // the first observed island.  Expose only the prefix whose expected
        // duration is plausibly covered by the observed speech mass.
        float CumExpected = 0.0f;
        PhoneLimit = 0;
        for (int32 PhoneIndex = 0; PhoneIndex < PhoneCount; ++PhoneIndex)
        {
            const FOffgridAIExpectedPhone& Expected = Plan.ExpectedPhones[PhoneIndex];
            const EOffgridAIPhoneClass C = ClassForPhoneBase(Expected.BasePhone);
            CumExpected += ExpectedDurationForPhone(Expected, C);
            if (CumExpected <= ObservedSpeechSeconds * 1.18f + 0.050f)
            {
                PhoneLimit = PhoneIndex + 1;
            }
            else
            {
                break;
            }
        }
    }

    PhoneLimit = FMath::Clamp(PhoneLimit, 0, PhoneCount);
    if (PhoneLimit <= 0)
    {
        return Out;
    }

    // If the feature stream is too under-resolved to assign at least one speech
    // frame per visible phone, final drain falls back to text-duration timing.
    // During streaming we wait for more evidence instead of fabricating phones.
    if (M < PhoneLimit)
    {
        if (Input.bFinal)
        {
            FillDurationFallback(Input, Out);
        }
        return Out;
    }

    FOffgridAITextVisemePlan VisiblePlan = Plan;
    if (PhoneLimit < PhoneCount)
    {
        VisiblePlan.ExpectedPhones.SetNum(PhoneLimit);
    }

    float VisibleExpectedSeconds = 0.0f;
    for (int32 PhoneIndex = 0; PhoneIndex < PhoneLimit; ++PhoneIndex)
    {
        const FOffgridAIExpectedPhone& Expected = Plan.ExpectedPhones[PhoneIndex];
        VisibleExpectedSeconds += ExpectedDurationForPhone(Expected, ClassForPhoneBase(Expected.BasePhone));
    }
    const float AcousticSpeechRateScale = VisibleExpectedSeconds > 0.010f
        ? FMath::Clamp(ObservedSpeechSeconds / VisibleExpectedSeconds, 0.80f, 1.30f)
        : 1.0f;

    // v17 monotonic-warp duration prior.  The runtime carries a conservative
    // canonical->observed time warp.  A visual rate below 1.0 means future
    // visemes should be stretched, which corresponds to a larger duration
    // scale here.  Local acoustic evidence and monotonic DP still own the
    // alignment; the warp only biases expected phone durations.
    const float WarpVisualRate = FMath::Clamp(
        Input.TimingWarpConfidence > 0.001f ? Input.TimingWarpRate : Input.TimingPLLPlayRate,
        0.88f, 1.12f);
    const float WarpConfidence = FMath::Clamp(FMath::Max(Input.TimingWarpConfidence, Input.TimingPLLConfidence), 0.0f, 1.0f);
    const float WarpDurationScale = FMath::Clamp(1.0f / WarpVisualRate, 0.86f, 1.16f);
    const float WarpBlend = FMath::Clamp(WarpConfidence * 0.55f, 0.0f, 0.55f);
    const float SpeechRateScale = FMath::Clamp(
        AcousticSpeechRateScale * (1.0f - WarpBlend) + WarpDurationScale * WarpBlend,
        0.80f,
        1.30f);
    Out.VisibleExpectedSeconds = VisibleExpectedSeconds;
    Out.SpeechRateScale = SpeechRateScale;

    TArray<FPhoneModel> Models;
    BuildPhoneModels(VisiblePlan, Frames, SpeechRateScale, Models);

    const int32 N = PhoneLimit;

    TArray<TArray<FDPCell>> DP;
    DP.SetNum(N);
    for (int32 PhoneIndex = 0; PhoneIndex < N; ++PhoneIndex)
    {
        DP[PhoneIndex].SetNum(M);
    }

    for (int32 PhoneIndex = 0; PhoneIndex < N; ++PhoneIndex)
    {
        const FPhoneModel& Model = Models[PhoneIndex];
        const int32 PhonesRemaining = N - PhoneIndex - 1;

        for (int32 EndFrame = PhoneIndex; EndFrame < M - PhonesRemaining; ++EndFrame)
        {
            const int32 MinStart = PhoneIndex == 0 ? 0 : PhoneIndex;
            const int32 MaxLen = FMath::Max(Model.MaxFrames, Model.MinFrames);
            const int32 EarliestStart = FMath::Max(MinStart, EndFrame - MaxLen + 1);
            const int32 LatestStart = EndFrame - Model.MinFrames + 1;
            if (LatestStart < EarliestStart)
            {
                continue;
            }

            for (int32 StartFrame = EarliestStart; StartFrame <= LatestStart; ++StartFrame)
            {
                const int32 PrevEnd = StartFrame - 1;
                float PrevScore = 0.0f;
                if (PhoneIndex > 0)
                {
                    if (!DP[PhoneIndex - 1].IsValidIndex(PrevEnd) || DP[PhoneIndex - 1][PrevEnd].Score <= -FLT_MAX * 0.5f)
                    {
                        continue;
                    }
                    PrevScore = DP[PhoneIndex - 1][PrevEnd].Score;
                }
                else if (StartFrame != 0)
                {
                    continue;
                }

                const bool bCrossesExpectedWordBoundary = PhoneIndex > 0
                    && VisiblePlan.ExpectedPhones.IsValidIndex(PhoneIndex - 1)
                    && VisiblePlan.ExpectedPhones.IsValidIndex(PhoneIndex)
                    && VisiblePlan.ExpectedPhones[PhoneIndex - 1].WordIndex != VisiblePlan.ExpectedPhones[PhoneIndex].WordIndex;
                const float BoundaryReward = PhoneIndex > 0
                    ? WordBoundaryTransitionReward(Frames, StartFrame, bCrossesExpectedWordBoundary)
                    : 0.0f;
                const float Candidate = PrevScore + SegmentScore(Frames, StartFrame, EndFrame, Model) + BoundaryReward;
                if (Candidate > DP[PhoneIndex][EndFrame].Score)
                {
                    DP[PhoneIndex][EndFrame].Score = Candidate;
                    DP[PhoneIndex][EndFrame].PrevFrame = PrevEnd;
                }
            }
        }
    }

    // The visible phone prefix is chosen from observed speech duration, so the
    // best path should consume the visible speech frames instead of leaving a
    // tail unused.  If exact full-consumption is impossible due to duration
    // constraints, fall back to the best reachable end frame.
    int32 BestEnd = M - 1;
    float BestScore = DP[N - 1][BestEnd].Score;
    if (BestScore <= -FLT_MAX * 0.5f)
    {
        BestEnd = INDEX_NONE;
        BestScore = -FLT_MAX;
        for (int32 EndFrame = N - 1; EndFrame < M; ++EndFrame)
        {
            const float Score = DP[N - 1][EndFrame].Score;
            if (Score > BestScore)
            {
                BestScore = Score;
                BestEnd = EndFrame;
            }
        }
    }

    if (BestEnd == INDEX_NONE || BestScore <= -FLT_MAX * 0.5f)
    {
        if (Input.bFinal)
        {
            FillDurationFallback(Input, Out);
        }
        return Out;
    }

    TArray<int32> StartFrames;
    TArray<int32> EndFrames;
    StartFrames.Init(INDEX_NONE, N);
    EndFrames.Init(INDEX_NONE, N);

    int32 EndFrame = BestEnd;
    for (int32 PhoneIndex = N - 1; PhoneIndex >= 0; --PhoneIndex)
    {
        const int32 PrevEnd = DP[PhoneIndex][EndFrame].PrevFrame;
        const int32 StartFrame = PrevEnd + 1;
        StartFrames[PhoneIndex] = StartFrame;
        EndFrames[PhoneIndex] = EndFrame;
        EndFrame = PrevEnd;
    }

    for (int32 PhoneIndex = 0; PhoneIndex < N; ++PhoneIndex)
    {
        const int32 S = StartFrames[PhoneIndex];
        const int32 E = EndFrames[PhoneIndex];
        if (!Frames.IsValidIndex(S) || !Frames.IsValidIndex(E))
        {
            continue;
        }

        const float PhoneStart = Frames[S].Start;
        const float PhoneEnd = Frames[E].End;
        const float PhoneCenter = (PhoneStart + PhoneEnd) * 0.5f;

        if (!Input.bFinal && PhoneEnd > StableEnd + 0.001f)
        {
            break;
        }

        const float Match = SegmentEmissionScore(Frames, S, E, Models[PhoneIndex].Class);
        const float Alternate = SegmentBestAlternateEmissionScore(Frames, S, E, Models[PhoneIndex].Class);
        const float Duration = FMath::Max(PhoneEnd - PhoneStart, 0.0f);
        Out.PhoneStartSeconds[PhoneIndex] = PhoneStart;
        Out.PhoneEndSeconds[PhoneIndex] = PhoneEnd;
        Out.PhoneCenterSeconds[PhoneIndex] = PhoneCenter;
        Out.PhoneMatchScores[PhoneIndex] = Match;
        Out.PhoneScoreGaps[PhoneIndex] = Match - Alternate;
        Out.PhoneObservedDurations[PhoneIndex] = Duration;
        Out.PhoneExpectedDurations[PhoneIndex] = Models[PhoneIndex].ExpectedDuration;
        Out.PhoneWordBoundarySalience[PhoneIndex] = PhoneIndex > 0 ? LocalBoundarySalience(Frames, S) : 0.0f;
        const bool bCrossesWordBoundary = PhoneIndex > 0 && Plan.ExpectedPhones[PhoneIndex - 1].WordIndex != Plan.ExpectedPhones[PhoneIndex].WordIndex;
        Out.PhoneAdvanceReasons[PhoneIndex] = (bCrossesWordBoundary && Out.PhoneWordBoundarySalience[PhoneIndex] >= 0.24f)
            ? FName(TEXT("streaming_forced_alignment_boundary_prior"))
            : FName(TEXT("streaming_forced_alignment_viterbi"));
        Out.HighestAlignedPhoneIndex = PhoneIndex;
        Out.HighestAlignedWordIndex = Plan.ExpectedPhones[PhoneIndex].WordIndex;
    }

    if (Plan.ExpectedPhones.IsValidIndex(Out.HighestAlignedPhoneIndex + 1))
    {
        Out.CurrentExpectedClass = ClassForPhoneBase(Plan.ExpectedPhones[Out.HighestAlignedPhoneIndex + 1].BasePhone);
    }
    else if (Plan.ExpectedPhones.IsValidIndex(Out.HighestAlignedPhoneIndex))
    {
        Out.CurrentExpectedClass = ClassForPhoneBase(Plan.ExpectedPhones[Out.HighestAlignedPhoneIndex].BasePhone);
    }
    Out.CurrentExpectedClassScore = Out.HighestAlignedPhoneIndex != INDEX_NONE && Out.PhoneMatchScores.IsValidIndex(Out.HighestAlignedPhoneIndex)
        ? Out.PhoneMatchScores[Out.HighestAlignedPhoneIndex]
        : 0.0f;
    Out.AdvanceReason = FName(TEXT("streaming_forced_alignment"));
    return Out;
}
