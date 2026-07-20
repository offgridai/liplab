#include "Lipsync/OffgridAIStreamingEvidenceSurface.h"

#include "Lipsync/OffgridAIAcousticEvidence.h"

#include <cmath>

namespace
{
#include "OffgridAIVowelNucleusModel.inl"
#include "OffgridAINucleusBeatModel.inl"

struct FLandmarkChannel
{
    EOffgridAIAudioLandmarkType Type;
    float Threshold;
    float MinSpacingSec;
};

static bool ChannelsCompete(
    EOffgridAIAudioLandmarkType Left,
    EOffgridAIAudioLandmarkType Right)
{
    // FV and sibilants share high-frequency frication evidence and need an
    // explicit margin. The other families are transcript-conditioned and
    // acoustically independent enough that cross-family vetoes discard useful
    // candidates without resolving a real ambiguity.
    const bool bLeftFricative = Left == EOffgridAIAudioLandmarkType::Labiodental
        || Left == EOffgridAIAudioLandmarkType::Sibilant;
    const bool bRightFricative = Right == EOffgridAIAudioLandmarkType::Labiodental
        || Right == EOffgridAIAudioLandmarkType::Sibilant;
    return bLeftFricative && bRightFricative;
}

static float Clamp01(float Value)
{
    return FMath::Clamp(Value, 0.0f, 1.0f);
}

static float Logistic(float Value)
{
    return 1.0f / (1.0f + static_cast<float>(std::exp(-FMath::Clamp(Value, -20.0f, 20.0f))));
}

static float Percentile(TArray<float> Values, float Quantile)
{
    if (Values.Num() <= 0) return 0.0f;
    Values.Sort([](float A, float B) { return A < B; });
    const int32 Index = FMath::Clamp(
        FMath::RoundToInt(Quantile * static_cast<float>(Values.Num() - 1)),
        0,
        Values.Num() - 1);
    return Values[Index];
}

static float ChannelScore(
    EOffgridAIAudioLandmarkType Type,
    const FOffgridAIStreamingAudioFeatureFrame& Frame,
    bool bUseRichFeatures)
{
    const FOffgridAIArticulatoryProbabilityField Field =
        FOffgridAIAcousticEvidence::BuildArticulatoryProbabilityField(Frame);
    const float BaseMBP = FMath::Max(
        Field.PhoneScores.Bilabial,
        FMath::Max(Field.Closure * 0.70f, Field.Release * 0.35f));
    const float BaseFV = FMath::Max(Field.PhoneScores.Labiodental, Field.Fricative * 0.85f);
    const float BaseGlide = FMath::Max(
        FMath::Max(Field.PhoneScores.Glide, Field.PhoneScores.Liquid * 0.50f),
        FMath::Max(Field.PhoneScores.VowelRound * 0.85f, Field.Sonorant * 0.60f));
    const float BaseSibilant = FMath::Max(Field.PhoneScores.Sibilant, Field.PhoneScores.StopBurst * 0.55f);
    const float BaseRound = FMath::Max(Field.PhoneScores.VowelRound, Field.Vowel * 0.50f);
    switch (Type)
    {
    case EOffgridAIAudioLandmarkType::Bilabial:
        if (bUseRichFeatures)
        {
            return Logistic(
                1.6154f
                + BaseMBP * 2.6109f
                - BaseFV * 5.7234f
                - BaseGlide * 0.9770f
                - BaseSibilant * 0.3313f
                - BaseRound * 0.7240f
                + Frame.RichLowBandNorm * 0.5200f
                - Frame.RichMidBandNorm * 0.7083f
                - Frame.RichHighBandNorm * 0.3794f
                - Frame.RichSpectralCentroidNorm * 1.8959f
                - Frame.RichSpectralRolloffNorm * 1.3588f
                + Frame.RichSpectralFlatness * 2.5045f
                - Frame.RichSpectralFlux * 0.0699f
                + Frame.RichPeriodicity * 0.6574f);
        }
        return Clamp01(Field.PhoneScores.Bilabial * 0.72f + Field.Closure * 0.18f + Field.Release * 0.10f);
    case EOffgridAIAudioLandmarkType::Labiodental:
        if (!bUseRichFeatures)
        {
            return Clamp01(Field.PhoneScores.Labiodental * 0.70f + Field.Fricative * 0.22f + Frame.HighBandNorm * 0.08f);
        }
        return Logistic(
            -1.5108f
            + BaseFV * 0.0326f
            - BaseSibilant * 0.1112f
            + Frame.RichLowBandNorm * 0.3115f
            - Frame.RichMidBandNorm * 0.7713f
            - Frame.RichHighBandNorm * 0.0827f
            - Frame.RichSpectralCentroidNorm * 0.0820f
            + Frame.RichSpectralRolloffNorm * 0.8287f
            + Frame.RichSpectralFlatness * 7.1829f
            - Frame.RichSpectralFlux * 0.0241f
            + Frame.RichPeriodicity * 0.5622f);
    case EOffgridAIAudioLandmarkType::Glide:
        if (bUseRichFeatures)
        {
            return Logistic(
                0.7559f
                + BaseMBP * 0.5479f
                - BaseFV * 4.6982f
                - BaseGlide * 0.0027f
                - BaseSibilant * 0.1409f
                + BaseRound * 0.2108f
                + Frame.RichLowBandNorm * 0.6384f
                - Frame.RichMidBandNorm * 1.7981f
                + Frame.RichHighBandNorm * 0.0126f
                - Frame.RichSpectralCentroidNorm * 1.2001f
                - Frame.RichSpectralRolloffNorm * 0.5835f
                - Frame.RichSpectralFlatness * 0.1318f
                + Frame.RichSpectralFlux * 0.3543f
                + Frame.RichPeriodicity * 0.3258f);
        }
        return Clamp01(Field.PhoneScores.Glide * 0.58f + Field.Sonorant * 0.22f + Field.PhoneScores.VowelRound * 0.20f);
    case EOffgridAIAudioLandmarkType::Sibilant:
        if (!bUseRichFeatures)
        {
            return Clamp01(Field.PhoneScores.Sibilant * 0.76f + Field.Fricative * 0.16f + Frame.HighBandNorm * 0.08f);
        }
        return Logistic(
            -1.4866f
            + BaseFV * 1.2480f
            + BaseSibilant * 2.1876f
            - Frame.RichLowBandNorm * 0.5052f
            - Frame.RichMidBandNorm * 1.5105f
            + Frame.RichHighBandNorm * 1.2470f
            + Frame.RichSpectralCentroidNorm * 2.2164f
            + Frame.RichSpectralRolloffNorm * 1.3184f
            - Frame.RichSpectralFlatness * 1.6500f
            + Frame.RichSpectralFlux * 0.1577f
            - Frame.RichPeriodicity * 0.4530f);
    case EOffgridAIAudioLandmarkType::RoundedVowel:
        if (bUseRichFeatures)
        {
            return Logistic(
                -0.9967f
                + BaseMBP * 0.3636f
                - BaseFV * 0.2654f
                + BaseGlide * 0.4834f
                - BaseSibilant * 0.8119f
                + BaseRound * 0.3470f
                + Frame.RichLowBandNorm * 0.2522f
                + Frame.RichMidBandNorm * 0.3160f
                - Frame.RichHighBandNorm * 0.5488f
                - Frame.RichSpectralCentroidNorm * 1.3194f
                - Frame.RichSpectralRolloffNorm * 1.1581f
                - Frame.RichSpectralFlatness * 2.7698f
                - Frame.RichSpectralFlux * 0.0819f
                + Frame.RichPeriodicity * 0.7846f);
        }
        return Clamp01(Field.PhoneScores.VowelRound * 0.72f + Field.Vowel * 0.18f + Field.Voiced * 0.10f);
    default:
        return 0.0f;
    }
}

static bool IsLocalMaximum(const TArray<float>& Scores, int32 Index, int32 Radius)
{
    if (!Scores.IsValidIndex(Index)) return false;
    for (int32 Offset = -Radius; Offset <= Radius; ++Offset)
    {
        if (Offset == 0 || !Scores.IsValidIndex(Index + Offset)) continue;
        if (Scores[Index + Offset] > Scores[Index]) return false;
    }
    return true;
}

static void AddObservation(
    TArray<FOffgridAIAudioLandmarkObservation>& Out,
    EOffgridAIAudioLandmarkType Type,
    const FOffgridAIStreamingAudioFeatureFrame& Frame,
    float Score,
    float DecisionSec)
{
    FOffgridAIAudioLandmarkObservation Observation;
    Observation.Type = Type;
    Observation.StartSec = Frame.AudioBufferStartSec;
    Observation.EndSec = Frame.AudioBufferEndSec;
    Observation.CenterSec = Frame.AudioBufferCenterSec;
    Observation.DecisionSec = FMath::Max(DecisionSec, Observation.CenterSec);
    Observation.Score = Clamp01(Score);
    Out.Add(Observation);
}

static void FrameVowelFeatures(
    const FOffgridAIStreamingAudioFeatureFrame& Frame,
    TArray<float>& Out)
{
    const FOffgridAIArticulatoryProbabilityField Field =
        FOffgridAIAcousticEvidence::BuildArticulatoryProbabilityField(Frame);
    Out = {
        Field.PhoneScores.VowelOpen,
        Field.PhoneScores.VowelFront,
        Field.PhoneScores.VowelRound,
        Field.Vowel,
        Field.Voiced,
        Field.Sonorant,
        Field.LowTilt,
        Field.HighTilt,
        Field.SpectralChange,
        Field.EnergyChange,
        Frame.RichLowBandNorm,
        Frame.RichMidBandNorm,
        Frame.RichHighBandNorm,
        Frame.RichSpectralCentroidNorm,
        Frame.RichSpectralRolloffNorm,
        Frame.RichSpectralFlatness,
        Frame.RichSpectralFlux,
        Frame.RichPeriodicity,
        Frame.RMSNorm,
        Frame.Periodicity,
        Frame.LowBandNorm,
        Frame.MidBandNorm,
        Frame.HighBandNorm,
        Frame.SpectralCentroidNorm
    };
}

static bool PulseVowelProbabilities(
    const TArray<FOffgridAIStreamingAudioFeatureFrame>& Frames,
    int32 PulseIndex,
    float OutProbabilities[3])
{
    constexpr int32 BaseFeatureCount = 24;
    constexpr int32 BandCount = 12;
    TArray<TArray<float>> Window;
    for (int32 Index = PulseIndex - 4; Index <= PulseIndex + 4; ++Index)
    {
        if (!Frames.IsValidIndex(Index)) continue;
        TArray<float> Features;
        FrameVowelFeatures(Frames[Index], Features);
        Window.Add(MoveTemp(Features));
    }
    if (Window.Num() <= 0) return false;

    TArray<float> ModelFeatures;
    for (int32 Aggregate = 0; Aggregate < 4; ++Aggregate)
    {
        for (int32 Feature = 0; Feature < BaseFeatureCount; ++Feature)
        {
            float Sum = 0.0f;
            float Maximum = -FLT_MAX;
            float Minimum = FLT_MAX;
            for (const TArray<float>& Row : Window)
            {
                Sum += Row[Feature];
                Maximum = FMath::Max(Maximum, Row[Feature]);
                Minimum = FMath::Min(Minimum, Row[Feature]);
            }
            if (Aggregate == 0) ModelFeatures.Add(Sum / static_cast<float>(Window.Num()));
            else if (Aggregate == 1) ModelFeatures.Add(Maximum);
            else if (Aggregate == 2) ModelFeatures.Add(Minimum);
            else ModelFeatures.Add(Window.Last()[Feature] - Window[0][Feature]);
        }
    }

    const int32 SpectralOffsets[] = { -3, -1, 1, 3 };
    TArray<const TArray<float>*> SpectralRows;
    for (const int32 Offset : SpectralOffsets)
    {
        const int32 Index = FMath::Clamp(PulseIndex + Offset, 0, Frames.Num() - 1);
        if (Frames[Index].RichBandDistribution.Num() != BandCount) return false;
        SpectralRows.Add(&Frames[Index].RichBandDistribution);
    }
    for (int32 Aggregate = 0; Aggregate < 4; ++Aggregate)
    {
        for (int32 Feature = 0; Feature < BandCount; ++Feature)
        {
            float Sum = 0.0f;
            float Maximum = -FLT_MAX;
            float Minimum = FLT_MAX;
            for (const TArray<float>* Row : SpectralRows)
            {
                Sum += (*Row)[Feature];
                Maximum = FMath::Max(Maximum, (*Row)[Feature]);
                Minimum = FMath::Min(Minimum, (*Row)[Feature]);
            }
            if (Aggregate == 0) ModelFeatures.Add(Sum / static_cast<float>(SpectralRows.Num()));
            else if (Aggregate == 1) ModelFeatures.Add(Maximum);
            else if (Aggregate == 2) ModelFeatures.Add(Minimum);
            else ModelFeatures.Add((*SpectralRows.Last())[Feature] - (*SpectralRows[0])[Feature]);
        }
    }
    if (ModelFeatures.Num() != GOffgridAIVowelFeatureCount) return false;

    float Logits[3] = { 0.0f, 0.0f, 0.0f };
    float MaxLogit = -FLT_MAX;
    for (int32 Class = 0; Class < 3; ++Class)
    {
        Logits[Class] = GOffgridAIVowelIntercepts[Class];
        for (int32 Feature = 0; Feature < ModelFeatures.Num(); ++Feature)
            Logits[Class] += GOffgridAIVowelWeights[Class][Feature] * ModelFeatures[Feature];
        MaxLogit = FMath::Max(MaxLogit, Logits[Class]);
    }
    float Sum = 0.0f;
    for (int32 Class = 0; Class < 3; ++Class)
    {
        OutProbabilities[Class] = static_cast<float>(std::exp(Logits[Class] - MaxLogit));
        Sum += OutProbabilities[Class];
    }
    for (int32 Class = 0; Class < 3; ++Class)
        OutProbabilities[Class] /= FMath::Max(Sum, 1.0e-6f);
    return true;
}

static float NucleusBeatProbability(
    const TArray<FOffgridAIStreamingAudioFeatureFrame>& Frames,
    const TArray<float>& Envelope2,
    const TArray<float>& Envelope3,
    const TArray<float>& Envelope5,
    const TArray<float>& Envelope8,
    int32 PulseIndex)
{
    auto Depth = [&](const TArray<float>& Envelope, int32 Context, int32 Inner)
    {
        if (!Envelope.IsValidIndex(PulseIndex - Context)
            || !Envelope.IsValidIndex(PulseIndex + Context)
            || Envelope[PulseIndex] < Envelope[PulseIndex - 1]
            || Envelope[PulseIndex] < Envelope[PulseIndex + 1])
        {
            return 0.0f;
        }
        float LeftFloor = Envelope[PulseIndex];
        float RightFloor = Envelope[PulseIndex];
        for (int32 Index = PulseIndex - Context; Index <= PulseIndex - Inner; ++Index)
            LeftFloor = FMath::Min(LeftFloor, Envelope[Index]);
        for (int32 Index = PulseIndex + Inner; Index <= PulseIndex + Context; ++Index)
            RightFloor = FMath::Min(RightFloor, Envelope[Index]);
        return FMath::Min(
            Envelope[PulseIndex] - LeftFloor,
            Envelope[PulseIndex] - RightFloor);
    };

    const FOffgridAIStreamingAudioFeatureFrame& Frame = Frames[PulseIndex];
    const FOffgridAIArticulatoryProbabilityField Field =
        FOffgridAIAcousticEvidence::BuildArticulatoryProbabilityField(Frame);
    const float PulseEnergy = Frame.RMSNorm
        * (0.45f + Field.Vowel * 0.40f + Frame.Periodicity * 0.15f);

    TArray<float> Features;
    const float BaseFeatures[] = {
        Depth(Envelope2, 10, 2),
        Depth(Envelope3, 10, 3),
        Depth(Envelope5, 10, 5),
        Depth(Envelope8, 10, 8),
        Depth(Envelope3, 6, 3),
        Depth(Envelope5, 14, 5),
        PulseEnergy,
        Frame.RMSNorm,
        Frame.SpeechEvidence,
        Frame.Periodicity,
        Field.Vowel,
        Field.Voiced,
        Field.Sonorant,
        Field.Fricative,
        Frame.ZCR,
        Frame.Flux,
        Field.Closure,
        Field.Transition,
        Field.SpectralChange,
        Field.EnergyChange,
        Frame.LowBandNorm,
        Frame.MidBandNorm,
        Frame.HighBandNorm
    };
    for (const float Value : BaseFeatures) Features.Add(Value);

    TArray<TArray<float>> Window;
    for (int32 Index = PulseIndex - 4; Index <= PulseIndex + 4; ++Index)
    {
        const FOffgridAIStreamingAudioFeatureFrame& WindowFrame = Frames[Index];
        const FOffgridAIArticulatoryProbabilityField WindowField =
            FOffgridAIAcousticEvidence::BuildArticulatoryProbabilityField(WindowFrame);
        Window.Add({
            WindowFrame.RichLowBandNorm,
            WindowFrame.RichMidBandNorm,
            WindowFrame.RichHighBandNorm,
            WindowFrame.RichSpectralCentroidNorm,
            WindowFrame.RichSpectralRolloffNorm,
            WindowFrame.RichSpectralFlatness,
            WindowFrame.RichSpectralFlux,
            WindowFrame.RichPeriodicity,
            WindowFrame.RMSNorm,
            WindowFrame.SpeechEvidence,
            WindowFrame.Periodicity,
            WindowField.Vowel,
            WindowField.Voiced,
            WindowField.Fricative,
            WindowFrame.ZCR
        });
    }
    for (int32 Aggregate = 0; Aggregate < 3; ++Aggregate)
    {
        for (int32 FeatureIndex = 0; FeatureIndex < Window[0].Num(); ++FeatureIndex)
        {
            float Sum = 0.0f;
            float Maximum = -FLT_MAX;
            float Minimum = FLT_MAX;
            for (const TArray<float>& Row : Window)
            {
                Sum += Row[FeatureIndex];
                Maximum = FMath::Max(Maximum, Row[FeatureIndex]);
                Minimum = FMath::Min(Minimum, Row[FeatureIndex]);
            }
            Features.Add(Aggregate == 0
                ? Sum / static_cast<float>(Window.Num())
                : (Aggregate == 1 ? Maximum : Minimum));
        }
    }
    for (const TArray<float>& Row : Window)
        for (const float Value : Row) Features.Add(Value);
    if (Features.Num() != GOffgridAINucleusBeatFeatureCount) return 0.0f;

    float Sum = 0.0f;
    for (int32 Tree = 0; Tree < GOffgridAINucleusBeatTreeCount; ++Tree)
    {
        int32 Node = GOffgridAINucleusBeatRoots[Tree];
        while (GOffgridAINucleusBeatFeatures[Node] >= 0)
        {
            const int32 FeatureIndex = GOffgridAINucleusBeatFeatures[Node];
            Node = Features[FeatureIndex] <= GOffgridAINucleusBeatThresholds[Node]
                ? GOffgridAINucleusBeatLeft[Node]
                : GOffgridAINucleusBeatRight[Node];
        }
        Sum += GOffgridAINucleusBeatProbability[Node];
    }
    return Sum / static_cast<float>(GOffgridAINucleusBeatTreeCount);
}
}

TArray<FOffgridAIAudioLandmarkObservation> FOffgridAIStreamingEvidenceSurface::Analyze(
    const TArray<FOffgridAIStreamingAudioFeatureFrame>& Frames,
    const FOffgridAIStreamingEvidenceSurfaceConfig& Config)
{
    TArray<FOffgridAIAudioLandmarkObservation> Out;
    if (Frames.Num() < 5) return Out;

    const float FrameSec = FMath::Max(
        Frames[0].AudioBufferEndSec - Frames[0].AudioBufferStartSec,
        0.001f);
    const int32 PostrollFrames = FMath::Max(FMath::RoundToInt(Config.PostrollSec / FrameSec), 1);
    const int32 PrerollFrames = FMath::Max(FMath::RoundToInt(Config.PrerollSec / FrameSec), 1);

    // Multi-scale envelope prominence is substantially more selective than raw
    // frame peaks. It requires a voiced speech pulse with valleys on both sides.
    TArray<float> PulseEnergy;
    for (const FOffgridAIStreamingAudioFeatureFrame& Frame : Frames)
    {
        const FOffgridAIArticulatoryProbabilityField Field =
            FOffgridAIAcousticEvidence::BuildArticulatoryProbabilityField(Frame);
        const float SonorityWeight = 0.45f + Field.Vowel * 0.40f + Frame.Periodicity * 0.15f;
        PulseEnergy.Add(Frame.RMSNorm * SonorityWeight);
    }
    TArray<float> VeryShortEnvelope;
    TArray<float> ShortEnvelope;
    TArray<float> MediumEnvelope;
    TArray<float> SlowEnvelope;
    auto BuildEnvelope = [&](int32 HalfWidth, TArray<float>& OutEnvelope)
    {
        for (int32 Index = 0; Index < Frames.Num(); ++Index)
        {
            float WeightedSum = 0.0f;
            float WeightSum = 0.0f;
            for (int32 Offset = -HalfWidth; Offset <= HalfWidth; ++Offset)
            {
                if (!Frames.IsValidIndex(Index + Offset)) continue;
                const float Weight = static_cast<float>(HalfWidth + 1 - FMath::Abs(Offset));
                WeightedSum += PulseEnergy[Index + Offset] * Weight;
                WeightSum += Weight;
            }
            OutEnvelope.Add(WeightSum > 0.0f ? WeightedSum / WeightSum : 0.0f);
        }
    };
    BuildEnvelope(2, VeryShortEnvelope);
    BuildEnvelope(3, ShortEnvelope);
    BuildEnvelope(5, MediumEnvelope);
    BuildEnvelope(8, SlowEnvelope);

    float LastPulseSec = -1000.0f;
    float LastPulseScore = 0.0f;
    int32 LastPulseOutputIndex = INDEX_NONE;
    int32 LastPulseFrameIndex = INDEX_NONE;
    int32 LastPulseEpoch = INDEX_NONE;
    const int32 PulseContextFrames = FMath::Min(
        14,
        PrerollFrames);
    for (int32 Index = PulseContextFrames; Index + PulseContextFrames < Frames.Num(); ++Index)
    {
        auto ScaleDepth = [&](const TArray<float>& Envelope, int32 InnerOffset, int32 ContextFrames)
        {
            const float Center = Envelope[Index];
            float LeftFloor = Center;
            float RightFloor = Center;
            for (int32 SampleIndex = Index - ContextFrames; SampleIndex <= Index - InnerOffset; ++SampleIndex)
            {
                LeftFloor = FMath::Min(LeftFloor, Envelope[SampleIndex]);
            }
            for (int32 SampleIndex = Index + InnerOffset; SampleIndex <= Index + ContextFrames; ++SampleIndex)
            {
                RightFloor = FMath::Min(RightFloor, Envelope[SampleIndex]);
            }
            const bool bPeak = Center >= Envelope[Index - 1] && Center >= Envelope[Index + 1];
            return bPeak ? FMath::Min(Center - LeftFloor, Center - RightFloor) : 0.0f;
        };
        const float VeryShortDepth = ScaleDepth(VeryShortEnvelope, 2, 10);
        const float ShortDepth = ScaleDepth(ShortEnvelope, 3, 10);
        const float MediumDepth = ScaleDepth(
            MediumEnvelope,
            5,
            10);
        const float SlowDepth = ScaleDepth(
            SlowEnvelope,
            8,
            10);
        float Prominence = FMath::Max(
            FMath::Max(VeryShortDepth, ShortDepth),
            FMath::Max(MediumDepth, SlowDepth));
        if (Prominence < 0.005f) continue;
        Prominence = NucleusBeatProbability(
            Frames,
            VeryShortEnvelope,
            ShortEnvelope,
            MediumEnvelope,
            SlowEnvelope,
            Index);
        if (Prominence < 0.375f) continue;
        int32 PulseEpoch = INDEX_NONE;
        if (Config.SpeechRegions)
        {
            for (int32 RegionIndex = 0; RegionIndex < Config.SpeechRegions->Num(); ++RegionIndex)
            {
                const FOffgridAIStreamingSpeechRegion& Region = (*Config.SpeechRegions)[RegionIndex];
                if (Frames[Index].AudioBufferCenterSec >= Region.AudioBufferStartSec - KINDA_SMALL_NUMBER
                    && Frames[Index].AudioBufferCenterSec <= Region.AudioBufferEndSec + KINDA_SMALL_NUMBER)
                {
                    PulseEpoch = RegionIndex;
                    break;
                }
            }
        }
        if (Config.SpeechRegions && PulseEpoch == INDEX_NONE)
            continue;
        if (LastPulseFrameIndex != INDEX_NONE && PulseEpoch != LastPulseEpoch)
        {
            LastPulseSec = -1000.0f;
            LastPulseScore = 0.0f;
            LastPulseOutputIndex = INDEX_NONE;
            LastPulseFrameIndex = INDEX_NONE;
        }
        const float PulseSpacingSec = Frames[Index].AudioBufferCenterSec - LastPulseSec;
        bool bDistinctClosePulse = false;
        if (LastPulseFrameIndex != INDEX_NONE
            && PulseSpacingSec >= 0.0f)
        {
            const TArray<float>& ValleyEnvelope = ShortEnvelope;
            float InterveningFloor = FMath::Min(
                ValleyEnvelope[LastPulseFrameIndex],
                ValleyEnvelope[Index]);
            for (int32 SampleIndex = LastPulseFrameIndex + 1; SampleIndex < Index; ++SampleIndex)
            {
                InterveningFloor = FMath::Min(InterveningFloor, ValleyEnvelope[SampleIndex]);
            }
            const float LowerPeak = FMath::Min(
                ValleyEnvelope[LastPulseFrameIndex],
                ValleyEnvelope[Index]);
            bDistinctClosePulse = LowerPeak - InterveningFloor
                >= 0.070f;
        }
        const float MinimumPulseSpacingSec = 0.120f;
        if (PulseSpacingSec < MinimumPulseSpacingSec && !bDistinctClosePulse)
        {
            if (Prominence > LastPulseScore && Out.IsValidIndex(LastPulseOutputIndex))
            {
                Out[LastPulseOutputIndex].StartSec = Frames[Index].AudioBufferStartSec;
                Out[LastPulseOutputIndex].EndSec = Frames[Index].AudioBufferEndSec;
                Out[LastPulseOutputIndex].CenterSec = Frames[Index].AudioBufferCenterSec;
                Out[LastPulseOutputIndex].DecisionSec = Frames[Index + PulseContextFrames].AudioBufferEndSec;
                Out[LastPulseOutputIndex].Score = Prominence;
                LastPulseSec = Frames[Index].AudioBufferCenterSec;
                LastPulseScore = Prominence;
                LastPulseFrameIndex = Index;
            }
            continue;
        }
        AddObservation(
            Out,
            EOffgridAIAudioLandmarkType::SyllabicPulse,
            Frames[Index],
            Prominence,
            Frames[Index + PulseContextFrames].AudioBufferEndSec);
        const int32 PulseOutputIndex = Out.Num() - 1;
        float VowelProbabilities[3];
        if (PulseVowelProbabilities(Frames, Index, VowelProbabilities))
        {
            const float DecisionSec = Frames[Index + PulseContextFrames].AudioBufferEndSec;
            const int32 BestClass = VowelProbabilities[1] > VowelProbabilities[0] ? 1 : 0;
            const int32 WinningClass = VowelProbabilities[2] > VowelProbabilities[BestClass] ? 2 : BestClass;
            if (VowelProbabilities[WinningClass] >= 0.50f && WinningClass != 2)
            {
                AddObservation(
                    Out,
                    WinningClass == 0
                        ? EOffgridAIAudioLandmarkType::OpenVowel
                        : EOffgridAIAudioLandmarkType::FrontVowel,
                    Frames[Index],
                    VowelProbabilities[WinningClass],
                    DecisionSec);
            }
        }
        LastPulseSec = Frames[Index].AudioBufferCenterSec;
        LastPulseScore = Prominence;
        LastPulseOutputIndex = PulseOutputIndex;
        LastPulseFrameIndex = Index;
        LastPulseEpoch = PulseEpoch;
    }

    // A missing onset pulse can cause a word to be stolen by the following
    // region, shifting the rest of the line. Recover only narrowly isolated
    // regions, and require either region closure or a later learned pulse as
    // confirmation. Search after 100 ms so an initial consonant burst cannot
    // masquerade as the syllable nucleus.
    if (Config.SpeechRegions && Config.SpeechRegions->Num() > 0)
    {
        for (int32 RegionIndex = 0;
            RegionIndex < Config.SpeechRegions->Num();
            ++RegionIndex)
        {
            const auto& Region = (*Config.SpeechRegions)[RegionIndex];
            const float RegionDurationSec =
                Region.AudioBufferEndSec - Region.AudioBufferStartSec;
            const bool bIsIsolatedOpeningWord = RegionIndex == 0
                && Region.bEnded
                && RegionDurationSec >= 0.450f
                && RegionDurationSec <= 0.600f;
            const float PrecedingGapSec = RegionIndex > 0
                ? Region.AudioBufferStartSec
                    - (*Config.SpeechRegions)[RegionIndex - 1].AudioBufferEndSec
                : 0.0f;
            const bool bIsIsolatedMediumRegion = RegionIndex > 0
                && PrecedingGapSec >= 0.280f
                && RegionDurationSec >= 0.650f
                && RegionDurationSec <= 0.800f;
            if (!bIsIsolatedOpeningWord && !bIsIsolatedMediumRegion)
                continue;
            const float SearchStartSec = Region.AudioBufferStartSec + 0.100f;
            const float SearchEndSec = FMath::Min(
                Region.AudioBufferStartSec + 0.300f,
                Region.bEnded ? Region.AudioBufferEndSec
                              : Region.AudioBufferStartSec + 0.300f);
            const bool bHasLaterPulse = std::any_of(
                Out.begin(), Out.end(), [&](const auto& Observation)
                {
                    return Observation.Type
                            == EOffgridAIAudioLandmarkType::SyllabicPulse
                        && Observation.CenterSec > SearchEndSec + 0.001f
                        && (!Region.bEnded
                            || Observation.CenterSec
                                <= Region.AudioBufferEndSec + 0.001f);
                });
            // Do not speculate merely because 300 ms elapsed. Backfill only
            // after a later learned pulse proves the region is syllabic, or
            // after the region closes with no learned pulse at all.
            const bool bReady = bHasLaterPulse || Region.bEnded;
            const bool bCovered = std::any_of(
                Out.begin(), Out.end(), [&](const auto& Observation)
                {
                    return Observation.Type
                            == EOffgridAIAudioLandmarkType::SyllabicPulse
                        && Observation.CenterSec
                            >= Region.AudioBufferStartSec - 0.001f
                        && Observation.CenterSec <= SearchEndSec + 0.001f;
                });
            if (bReady && !bCovered && SearchStartSec <= SearchEndSec)
            {
                int32 BestFrameIndex = INDEX_NONE;
                float BestEnergy = -1.0f;
                for (int32 FrameIndex = 0;
                    FrameIndex < Frames.Num();
                    ++FrameIndex)
                {
                    const auto& Frame = Frames[FrameIndex];
                    if (!Frame.bInSpeechAfterFrame
                        || Frame.AudioBufferCenterSec
                            < SearchStartSec - 0.001f
                        || Frame.AudioBufferCenterSec
                            > SearchEndSec + 0.001f)
                        continue;
                    if (PulseEnergy[FrameIndex] > BestEnergy)
                    {
                        BestEnergy = PulseEnergy[FrameIndex];
                        BestFrameIndex = FrameIndex;
                    }
                }
                if (BestFrameIndex != INDEX_NONE)
                {
                    const int32 DecisionIndex = FMath::Min(
                        BestFrameIndex + PulseContextFrames,
                        Frames.Num() - 1);
                    AddObservation(
                        Out,
                        EOffgridAIAudioLandmarkType::SyllabicPulse,
                        Frames[BestFrameIndex],
                        0.38f,
                        FMath::Max(
                            Frames[DecisionIndex].AudioBufferEndSec,
                            SearchEndSec));
                }
            }
        }
    }

    // Lull/resume observations are state transitions, not every quiet frame.
    bool bInLull = false;
    bool bLullEmitted = false;
    int32 LullStartIndex = INDEX_NONE;
    for (int32 Index = 0; Index < Frames.Num(); ++Index)
    {
        const auto& Frame = Frames[Index];
        const bool bConfirmedNucleus = std::any_of(
            Out.begin(), Out.end(), [&](const FOffgridAIAudioLandmarkObservation& Observation)
            {
                return Observation.Type == EOffgridAIAudioLandmarkType::SyllabicPulse
                    && FMath::Abs(Observation.CenterSec - Frame.AudioBufferCenterSec)
                        <= FrameSec * 0.51f;
            });
        const bool bQuiet = !bConfirmedNucleus && (Frame.bStrongQuiet
            || (Frame.SpeechEvidence <= 0.085f && Frame.RMSNorm <= 0.075f));
        if (!bInLull && bQuiet)
        {
            bInLull = true;
            bLullEmitted = false;
            LullStartIndex = Index;
        }
        else if (bInLull && bQuiet && !bLullEmitted && Index - LullStartIndex + 1 >= 3)
        {
            const auto& StartFrame = Frames[LullStartIndex];
            AddObservation(
                Out,
                EOffgridAIAudioLandmarkType::Lull,
                StartFrame,
                0.55f,
                Frame.AudioBufferEndSec);
            bLullEmitted = true;
        }
        else if (bInLull && !bQuiet)
        {
            if (bLullEmitted)
            {
                AddObservation(
                    Out,
                    EOffgridAIAudioLandmarkType::Resume,
                    Frame,
                    Clamp01(Frame.SpeechEvidence * 0.65f + Frame.Flux * 0.35f),
                    Frame.AudioBufferEndSec);
            }
            bInLull = false;
            bLullEmitted = false;
            LullStartIndex = INDEX_NONE;
        }
    }

    const FLandmarkChannel Channels[] = {
        // Only channels with useful corpus-level discrimination are emitted.
        // MBP, glide, and rounded-vowel evidence remains available in the
        // probability field for later transcript conditioning, but is too
        // ambiguous to publish as an unconditioned landmark.
        { EOffgridAIAudioLandmarkType::Bilabial,
            0.58f, 0.070f },
        { EOffgridAIAudioLandmarkType::Labiodental,
            0.48f, 0.070f },
        { EOffgridAIAudioLandmarkType::Glide,
            0.56f, 0.070f },
        { EOffgridAIAudioLandmarkType::Sibilant,
            0.56f, 0.070f },
        { EOffgridAIAudioLandmarkType::RoundedVowel,
            0.55f, 0.070f }
    };
    TArray<TArray<float>> ChannelScores;
    for (const FLandmarkChannel& Channel : Channels)
    {
        TArray<float> Scores;
        for (const auto& Frame : Frames)
        {
            Scores.Add(ChannelScore(Channel.Type, Frame, true));
        }
        {
            const TArray<float> RawScores = Scores;
            for (int32 Index = 0; Index < Scores.Num(); ++Index)
            {
                float Sum = 0.0f;
                int32 Count = 0;
                for (int32 Offset = -2; Offset <= 2; ++Offset)
                {
                    if (!RawScores.IsValidIndex(Index + Offset)) continue;
                    Sum += RawScores[Index + Offset];
                    ++Count;
                }
                const float SegmentMean = Count > 0 ? Sum / static_cast<float>(Count) : RawScores[Index];
                // Closures/releases are brief events; averaging them like a
                // sustained fricative erases the very transition MBP relies on.
                const float CenterWeight = Channel.Type == EOffgridAIAudioLandmarkType::Bilabial
                    ? 0.75f
                    : 0.40f;
                Scores[Index] = Clamp01(
                    RawScores[Index] * CenterWeight
                    + SegmentMean * (1.0f - CenterWeight));
            }
        }
        ChannelScores.Add(Scores);
    }

    for (int32 ChannelIndex = 0; ChannelIndex < static_cast<int32>(sizeof(Channels) / sizeof(Channels[0])); ++ChannelIndex)
    {
        const FLandmarkChannel& Channel = Channels[ChannelIndex];
        const TArray<float>& Scores = ChannelScores[ChannelIndex];

        float LastCenterSec = -1000.0f;
        float LastAssignedPulseCenterSec = -1000.0f;
        int32 LastChannelOutputIndex = INDEX_NONE;
        for (int32 Index = 2; Index + 2 < Frames.Num(); ++Index)
        {
            if (!Frames[Index].bInSpeechAfterFrame || !IsLocalMaximum(Scores, Index, 2)) continue;
            const int32 HistoryStart = FMath::Max(0, Index - PostrollFrames);
            TArray<float> HistoryScores;
            for (int32 HistoryIndex = HistoryStart; HistoryIndex < Index; ++HistoryIndex)
            {
                HistoryScores.Add(Scores[HistoryIndex]);
            }
            const float HistoricalMedian = Percentile(HistoryScores, 0.50f);
            const float Contrast = Clamp01((Scores[Index] - HistoricalMedian) / 0.24f);
            const float EventScore = Clamp01(Scores[Index] * 0.78f + Contrast * 0.22f);
            float BestCompetitor = 0.0f;
            for (int32 OtherChannel = 0; OtherChannel < ChannelScores.Num(); ++OtherChannel)
            {
                if (OtherChannel == ChannelIndex) continue;
                if (Channels[OtherChannel].Threshold > 1.0f) continue;
                if (!ChannelsCompete(Channel.Type, Channels[OtherChannel].Type)) continue;
                BestCompetitor = FMath::Max(BestCompetitor, ChannelScores[OtherChannel][Index]);
            }
            const float Margin = EventScore - BestCompetitor;
            if (EventScore < Channel.Threshold
                || Margin < 0.025f
                || Frames[Index].AudioBufferCenterSec - LastCenterSec < Channel.MinSpacingSec)
            {
                continue;
            }
            {
                float MaxPulseDistanceSec = 1000.0f;
                switch (Channel.Type)
                {
                case EOffgridAIAudioLandmarkType::Glide: MaxPulseDistanceSec = 0.120f; break;
                case EOffgridAIAudioLandmarkType::RoundedVowel: MaxPulseDistanceSec = 0.080f; break;
                default: break;
                }
                float NearestPulseDistanceSec = 1000.0f;
                float NearestPulseCenterSec = -1000.0f;
                for (const FOffgridAIAudioLandmarkObservation& Observation : Out)
                {
                    if (Observation.Type != EOffgridAIAudioLandmarkType::SyllabicPulse) continue;
                    const float Distance = FMath::Abs(
                        Frames[Index].AudioBufferCenterSec - Observation.CenterSec);
                    if (Distance < NearestPulseDistanceSec)
                    {
                        NearestPulseDistanceSec = Distance;
                        NearestPulseCenterSec = Observation.CenterSec;
                    }
                }
                if (NearestPulseCenterSec < 0.0f || NearestPulseDistanceSec > MaxPulseDistanceSec) continue;
                const bool bConsolidatePerPulse = true;
                if (bConsolidatePerPulse
                    && FMath::Abs(NearestPulseCenterSec - LastAssignedPulseCenterSec) < 0.001f
                    && Out.IsValidIndex(LastChannelOutputIndex))
                {
                    if (EventScore > Out[LastChannelOutputIndex].Score)
                    {
                        Out[LastChannelOutputIndex].StartSec = Frames[Index].AudioBufferStartSec;
                        Out[LastChannelOutputIndex].EndSec = Frames[Index].AudioBufferEndSec;
                        Out[LastChannelOutputIndex].CenterSec = Frames[Index].AudioBufferCenterSec;
                        Out[LastChannelOutputIndex].DecisionSec = Frames[FMath::Min(
                            Index + 2,
                            FMath::Min(Index + PrerollFrames, Frames.Num() - 1))].AudioBufferEndSec;
                        Out[LastChannelOutputIndex].Score = EventScore;
                        LastCenterSec = Frames[Index].AudioBufferCenterSec;
                    }
                    continue;
                }
                LastAssignedPulseCenterSec = NearestPulseCenterSec;
            }
            const int32 DecisionIndex = FMath::Min(Index + 2, FMath::Min(Index + PrerollFrames, Frames.Num() - 1));
            AddObservation(
                Out,
                Channel.Type,
                Frames[Index],
                EventScore,
                Frames[DecisionIndex].AudioBufferEndSec);
            LastChannelOutputIndex = Out.Num() - 1;
            LastCenterSec = Frames[Index].AudioBufferCenterSec;
        }
    }

    Out.Sort([](const auto& A, const auto& B)
    {
        if (A.CenterSec != B.CenterSec) return A.CenterSec < B.CenterSec;
        return static_cast<uint8>(A.Type) < static_cast<uint8>(B.Type);
    });
    return Out;
}

FString FOffgridAIStreamingEvidenceSurface::LandmarkTypeToString(EOffgridAIAudioLandmarkType Type)
{
    switch (Type)
    {
    case EOffgridAIAudioLandmarkType::SyllabicPulse: return TEXT("pulse");
    case EOffgridAIAudioLandmarkType::Lull: return TEXT("lull");
    case EOffgridAIAudioLandmarkType::Resume: return TEXT("resume");
    case EOffgridAIAudioLandmarkType::Bilabial: return TEXT("mbp");
    case EOffgridAIAudioLandmarkType::Labiodental: return TEXT("fv");
    case EOffgridAIAudioLandmarkType::Glide: return TEXT("glide");
    case EOffgridAIAudioLandmarkType::Sibilant: return TEXT("sibilant");
    case EOffgridAIAudioLandmarkType::OpenVowel: return TEXT("open");
    case EOffgridAIAudioLandmarkType::FrontVowel: return TEXT("front");
    case EOffgridAIAudioLandmarkType::RoundedVowel: return TEXT("round");
    default: return TEXT("unknown");
    }
}
