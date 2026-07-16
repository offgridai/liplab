#include "Lipsync/OffgridAIStreamingEvidenceSurface.h"

#include "Lipsync/OffgridAIAcousticEvidence.h"

#include <cmath>

namespace
{
#include "OffgridAIVowelNucleusModel.inl"

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
    BuildEnvelope(4, MediumEnvelope);
    BuildEnvelope(6, SlowEnvelope);

    float LastPulseSec = -1000.0f;
    float LastPulseScore = 0.0f;
    int32 LastPulseOutputIndex = INDEX_NONE;
    int32 LastPulseFrameIndex = INDEX_NONE;
    const int32 PulseContextFrames = FMath::Min(10, PrerollFrames);
    for (int32 Index = PulseContextFrames; Index + PulseContextFrames < Frames.Num(); ++Index)
    {
        auto ScaleProminence = [&](const TArray<float>& Envelope, int32 InnerOffset)
        {
            const float Center = Envelope[Index];
            float LeftFloor = Center;
            float RightFloor = Center;
            for (int32 SampleIndex = Index - PulseContextFrames; SampleIndex <= Index - InnerOffset; ++SampleIndex)
            {
                LeftFloor = FMath::Min(LeftFloor, Envelope[SampleIndex]);
            }
            for (int32 SampleIndex = Index + InnerOffset; SampleIndex <= Index + PulseContextFrames; ++SampleIndex)
            {
                RightFloor = FMath::Min(RightFloor, Envelope[SampleIndex]);
            }
            const bool bPeak = Center >= Envelope[Index - 1] && Center >= Envelope[Index + 1];
            return bPeak ? Clamp01(FMath::Min(Center - LeftFloor, Center - RightFloor) / 0.12f) : 0.0f;
        };
        float Prominence = FMath::Max(
            ScaleProminence(MediumEnvelope, 4),
            ScaleProminence(SlowEnvelope, 6) * 1.05f);
        const FOffgridAIArticulatoryProbabilityField Field =
            FOffgridAIAcousticEvidence::BuildArticulatoryProbabilityField(Frames[Index]);
        const float VowelSupport = Clamp01(Field.Vowel * 0.60f + Frames[Index].Periodicity * 0.40f);
        const float MinProminence = 0.10f;
        const float MinSpeechEvidence = 0.18f;
        const float MinVowelSupport = 0.35f;
        if (Prominence < MinProminence
            || Frames[Index].SpeechEvidence < MinSpeechEvidence
            || VowelSupport < MinVowelSupport)
        {
            continue;
        }
        const float PulseSpacingSec = Frames[Index].AudioBufferCenterSec - LastPulseSec;
        bool bDistinctClosePulse = false;
        if (LastPulseFrameIndex != INDEX_NONE
            && PulseSpacingSec >= 0.060f)
        {
            float InterveningFloor = FMath::Min(MediumEnvelope[LastPulseFrameIndex], MediumEnvelope[Index]);
            for (int32 SampleIndex = LastPulseFrameIndex + 1; SampleIndex < Index; ++SampleIndex)
            {
                InterveningFloor = FMath::Min(InterveningFloor, MediumEnvelope[SampleIndex]);
            }
            const float LowerPeak = FMath::Min(MediumEnvelope[LastPulseFrameIndex], MediumEnvelope[Index]);
            bDistinctClosePulse = LowerPeak - InterveningFloor >= 0.035f;
        }
        if (PulseSpacingSec < 0.100f && !bDistinctClosePulse)
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
    }

    // Lull/resume observations are state transitions, not every quiet frame.
    bool bInLull = false;
    bool bLullEmitted = false;
    int32 LullStartIndex = INDEX_NONE;
    for (int32 Index = 0; Index < Frames.Num(); ++Index)
    {
        const auto& Frame = Frames[Index];
        const bool bQuiet = Frame.bStrongQuiet
            || (Frame.SpeechEvidence <= 0.085f && Frame.RMSNorm <= 0.075f);
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
            Config.bPermissivePhoneCandidates ? 0.58f : 1.10f, 0.070f },
        { EOffgridAIAudioLandmarkType::Labiodental,
            Config.bPermissivePhoneCandidates ? 0.48f : 0.52f, 0.070f },
        { EOffgridAIAudioLandmarkType::Glide,
            Config.bPermissivePhoneCandidates ? 0.56f : 1.10f, 0.070f },
        { EOffgridAIAudioLandmarkType::Sibilant,
            Config.bPermissivePhoneCandidates ? 0.56f : 0.88f, 0.070f },
        { EOffgridAIAudioLandmarkType::RoundedVowel,
            Config.bPermissivePhoneCandidates ? 0.55f : 1.10f, 0.070f }
    };
    TArray<TArray<float>> ChannelScores;
    for (const FLandmarkChannel& Channel : Channels)
    {
        TArray<float> Scores;
        for (const auto& Frame : Frames)
        {
            Scores.Add(ChannelScore(Channel.Type, Frame, Config.bPermissivePhoneCandidates));
        }
        if (Config.bPermissivePhoneCandidates)
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
            if (Config.bPermissivePhoneCandidates)
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
