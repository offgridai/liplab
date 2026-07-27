#include "Lipsync/OffgridAIStreamingAudioFeatures.h"

#include <cmath>

void FOffgridAIStreamingAudioFeatureExtractor::Reset()
{
    FeatureFrames.Reset();
    PendingMonoSamples.Reset();
    RichAnalysisSamples.Reset();
    PreviousRichBandDistribution.Reset();
    PendingSampleBase = 0;
    ActiveSampleRate = 0;
    SpeechPeakRMS = 0.0001f;
    ObservedAudioBufferEndSec = 0.0f;
}

void FOffgridAIStreamingAudioFeatureExtractor::AppendPCM16(const TArray<uint8>& PCMChunk, int32 BytesToUse, int32 SampleRate, int32 NumChannels, int64 ChunkStartSample)
{
    if (BytesToUse <= 0 || SampleRate <= 0 || NumChannels <= 0)
    {
        return;
    }
    if (ActiveSampleRate != SampleRate)
    {
        PendingMonoSamples.Reset();
        RichAnalysisSamples.Reset();
        PreviousRichBandDistribution.Reset();
        PendingSampleBase = ChunkStartSample >= 0 ? ChunkStartSample : 0;
        ActiveSampleRate = SampleRate;
    }

    const int32 BytesPerFrame = NumChannels * static_cast<int32>(sizeof(int16));
    const int32 Frames = BytesPerFrame > 0 ? BytesToUse / BytesPerFrame : 0;
    const int16* Samples = reinterpret_cast<const int16*>(PCMChunk.GetData());
    const int32 Int16Count = BytesToUse / static_cast<int32>(sizeof(int16));
    for (int32 F = 0; F < Frames; ++F)
    {
        float Mono = 0.0f;
        for (int32 C = 0; C < NumChannels; ++C)
        {
            const int32 Index = F * NumChannels + C;
            if (Index < Int16Count)
            {
                Mono += static_cast<float>(Samples[Index]) / 32768.0f;
            }
        }
        PendingMonoSamples.Add(Mono / static_cast<float>(NumChannels));
    }

    const int32 Hop = FMath::Max(FMath::RoundToInt(0.010f * SampleRate), 1);
    while (PendingMonoSamples.Num() >= Hop)
    {
        double SumSq = 0.0;
        int32 Crossings = 0;
        float Prev = PendingMonoSamples[0];
        for (int32 I = 0; I < Hop; ++I)
        {
            const float S = PendingMonoSamples[I];
            SumSq += static_cast<double>(S) * static_cast<double>(S);
            if (I > 0 && ((Prev < 0.0f && S >= 0.0f) || (Prev >= 0.0f && S < 0.0f))) ++Crossings;
            Prev = S;
        }
        const float RMS = FMath::Sqrt(static_cast<float>(SumSq / static_cast<double>(Hop)));

        auto GoertzelEnergy = [&](float FrequencyHz) -> float
        {
            const float Nyquist = static_cast<float>(SampleRate) * 0.5f;
            if (FrequencyHz <= 0.0f || FrequencyHz >= Nyquist || Hop <= 2) return 0.0f;
            const float K = FMath::FloorToFloat(0.5f + (static_cast<float>(Hop) * FrequencyHz) / static_cast<float>(SampleRate));
            const float Omega = 2.0f * PI * K / static_cast<float>(Hop);
            const float Coeff = 2.0f * FMath::Cos(Omega);
            float Q0 = 0.0f;
            float Q1 = 0.0f;
            float Q2 = 0.0f;
            for (int32 I = 0; I < Hop; ++I)
            {
                const float Window = 0.54f - 0.46f * FMath::Cos(2.0f * PI * static_cast<float>(I) / static_cast<float>(FMath::Max(Hop - 1, 1)));
                Q0 = Coeff * Q1 - Q2 + PendingMonoSamples[I] * Window;
                Q2 = Q1;
                Q1 = Q0;
            }
            return FMath::Max(Q1 * Q1 + Q2 * Q2 - Coeff * Q1 * Q2, 0.0f);
        };

        const float E300 = GoertzelEnergy(300.0f);
        const float E700 = GoertzelEnergy(700.0f);
        const float E1200 = GoertzelEnergy(1200.0f);
        const float E2400 = GoertzelEnergy(2400.0f);
        const float E4000 = GoertzelEnergy(4000.0f);
        const float E6500 = GoertzelEnergy(6500.0f);
        const float LowEnergy = E300 + E700;
        const float MidEnergy = E1200 + E2400;
        const float HighEnergy = E4000 + E6500;
        const float TotalBandEnergy = FMath::Max(LowEnergy + MidEnergy + HighEnergy, 0.000001f);
        const float LowBandNorm = FMath::Clamp(LowEnergy / TotalBandEnergy, 0.0f, 1.0f);
        const float MidBandNorm = FMath::Clamp(MidEnergy / TotalBandEnergy, 0.0f, 1.0f);
        const float HighBandNorm = FMath::Clamp(HighEnergy / TotalBandEnergy, 0.0f, 1.0f);
        const float CentroidHz = (E300 * 300.0f + E700 * 700.0f + E1200 * 1200.0f + E2400 * 2400.0f + E4000 * 4000.0f + E6500 * 6500.0f) / TotalBandEnergy;
        const float SpectralCentroidNorm = FMath::Clamp(CentroidHz / FMath::Max(static_cast<float>(SampleRate) * 0.5f, 1.0f), 0.0f, 1.0f);

        float BestAutoCorr = 0.0f;
        const int32 MinLag = FMath::Max(FMath::RoundToInt(0.0025f * SampleRate), 1);
        const int32 MaxLag = FMath::Min(FMath::RoundToInt(0.0080f * SampleRate), Hop - 2);
        if (MaxLag > MinLag && SumSq > 0.0000001)
        {
            for (int32 Lag = MinLag; Lag <= MaxLag; Lag += FMath::Max((MaxLag - MinLag) / 6, 1))
            {
                double Corr = 0.0;
                double A = 0.0;
                double B = 0.0;
                for (int32 I = 0; I + Lag < Hop; ++I)
                {
                    const float X = PendingMonoSamples[I];
                    const float Y = PendingMonoSamples[I + Lag];
                    Corr += static_cast<double>(X) * static_cast<double>(Y);
                    A += static_cast<double>(X) * static_cast<double>(X);
                    B += static_cast<double>(Y) * static_cast<double>(Y);
                }
                const float NormCorr = static_cast<float>(Corr / FMath::Sqrt(FMath::Max(A * B, 0.00000001)));
                BestAutoCorr = FMath::Max(BestAutoCorr, NormCorr);
            }
        }
        const float Periodicity = FMath::Clamp(BestAutoCorr, 0.0f, 1.0f);

        for (int32 I = 0; I < Hop; ++I)
        {
            RichAnalysisSamples.Add(PendingMonoSamples[I]);
        }
        const int32 RichWindowSamples = FMath::Max(FMath::RoundToInt(0.030f * SampleRate), Hop);
        if (RichAnalysisSamples.Num() > RichWindowSamples)
        {
            RichAnalysisSamples.RemoveAt(0, RichAnalysisSamples.Num() - RichWindowSamples, EAllowShrinking::No);
        }

        float RichLowBandNorm = LowBandNorm;
        float RichMidBandNorm = MidBandNorm;
        float RichHighBandNorm = HighBandNorm;
        float RichCentroidNorm = SpectralCentroidNorm;
        float RichRolloffNorm = SpectralCentroidNorm;
        float RichFlatness = 0.0f;
        float RichFlux = 0.0f;
        float RichPeriodicity = Periodicity;
        if (RichAnalysisSamples.Num() >= RichWindowSamples)
        {
            auto RichGoertzelEnergy = [&](float FrequencyHz) -> float
            {
                const int32 N = RichAnalysisSamples.Num();
                const float Nyquist = static_cast<float>(SampleRate) * 0.5f;
                if (FrequencyHz <= 0.0f || FrequencyHz >= Nyquist || N <= 2) return 0.0f;
                const float K = FMath::FloorToFloat(0.5f + (static_cast<float>(N) * FrequencyHz) / static_cast<float>(SampleRate));
                const float Omega = 2.0f * PI * K / static_cast<float>(N);
                const float Coeff = 2.0f * FMath::Cos(Omega);
                float Q0 = 0.0f;
                float Q1 = 0.0f;
                float Q2 = 0.0f;
                for (int32 I = 0; I < N; ++I)
                {
                    const float Window = 0.54f - 0.46f * FMath::Cos(
                        2.0f * PI * static_cast<float>(I) / static_cast<float>(FMath::Max(N - 1, 1)));
                    Q0 = Coeff * Q1 - Q2 + RichAnalysisSamples[I] * Window;
                    Q2 = Q1;
                    Q1 = Q0;
                }
                return FMath::Max(Q1 * Q1 + Q2 * Q2 - Coeff * Q1 * Q2, 0.0f);
            };

            static const float FrequenciesHz[] = {
                250.0f, 400.0f, 630.0f, 1000.0f, 1400.0f, 2000.0f,
                2800.0f, 3600.0f, 4500.0f, 5400.0f, 6300.0f, 7200.0f
            };
            TArray<float> Energies;
            float TotalEnergy = 0.0f;
            float RichLowEnergy = 0.0f;
            float RichMidEnergy = 0.0f;
            float RichHighEnergy = 0.0f;
            float WeightedFrequency = 0.0f;
            double LogEnergySum = 0.0;
            for (float FrequencyHz : FrequenciesHz)
            {
                const float Energy = RichGoertzelEnergy(FrequencyHz);
                Energies.Add(Energy);
                TotalEnergy += Energy;
                WeightedFrequency += Energy * FrequencyHz;
                LogEnergySum += std::log(static_cast<double>(Energy) + 1.0e-9);
                if (FrequencyHz < 1000.0f) RichLowEnergy += Energy;
                else if (FrequencyHz < 3500.0f) RichMidEnergy += Energy;
                else RichHighEnergy += Energy;
            }
            TotalEnergy = FMath::Max(TotalEnergy, 0.000001f);
            RichLowBandNorm = FMath::Clamp(RichLowEnergy / TotalEnergy, 0.0f, 1.0f);
            RichMidBandNorm = FMath::Clamp(RichMidEnergy / TotalEnergy, 0.0f, 1.0f);
            RichHighBandNorm = FMath::Clamp(RichHighEnergy / TotalEnergy, 0.0f, 1.0f);
            RichCentroidNorm = FMath::Clamp(
                (WeightedFrequency / TotalEnergy) / FMath::Max(static_cast<float>(SampleRate) * 0.5f, 1.0f),
                0.0f,
                1.0f);
            const double GeometricMean = std::exp(LogEnergySum / static_cast<double>(Energies.Num()));
            const double ArithmeticMean = static_cast<double>(TotalEnergy) / static_cast<double>(Energies.Num());
            RichFlatness = FMath::Clamp(static_cast<float>(GeometricMean / FMath::Max(ArithmeticMean, 1.0e-9)), 0.0f, 1.0f);

            const float RolloffEnergy = TotalEnergy * 0.85f;
            float CumulativeEnergy = 0.0f;
            for (int32 I = 0; I < Energies.Num(); ++I)
            {
                CumulativeEnergy += Energies[I];
                if (CumulativeEnergy >= RolloffEnergy)
                {
                    RichRolloffNorm = FMath::Clamp(
                        FrequenciesHz[I] / FMath::Max(static_cast<float>(SampleRate) * 0.5f, 1.0f),
                        0.0f,
                        1.0f);
                    break;
                }
            }

            TArray<float> Distribution;
            for (float Energy : Energies) Distribution.Add(Energy / TotalEnergy);
            if (PreviousRichBandDistribution.Num() == Distribution.Num())
            {
                for (int32 I = 0; I < Distribution.Num(); ++I)
                {
                    RichFlux += FMath::Max(Distribution[I] - PreviousRichBandDistribution[I], 0.0f);
                }
                RichFlux = FMath::Clamp(RichFlux * 2.0f, 0.0f, 1.0f);
            }
            PreviousRichBandDistribution = Distribution;

            float BestRichAutoCorr = 0.0f;
            const int32 MinRichLag = FMath::Max(FMath::RoundToInt(0.0025f * SampleRate), 1);
            const int32 MaxRichLag = FMath::Min(FMath::RoundToInt(0.0125f * SampleRate), RichAnalysisSamples.Num() - 2);
            const int32 LagStep = FMath::Max((MaxRichLag - MinRichLag) / 24, 1);
            for (int32 Lag = MinRichLag; Lag <= MaxRichLag; Lag += LagStep)
            {
                double Corr = 0.0;
                double A = 0.0;
                double B = 0.0;
                for (int32 I = 0; I + Lag < RichAnalysisSamples.Num(); ++I)
                {
                    const float X = RichAnalysisSamples[I];
                    const float Y = RichAnalysisSamples[I + Lag];
                    Corr += static_cast<double>(X) * static_cast<double>(Y);
                    A += static_cast<double>(X) * static_cast<double>(X);
                    B += static_cast<double>(Y) * static_cast<double>(Y);
                }
                const float NormCorr = static_cast<float>(Corr / std::sqrt(FMath::Max(A * B, 1.0e-8)));
                BestRichAutoCorr = FMath::Max(BestRichAutoCorr, NormCorr);
            }
            RichPeriodicity = FMath::Clamp(BestRichAutoCorr, 0.0f, 1.0f);
        }

        const float Start = static_cast<float>(PendingSampleBase) / static_cast<float>(SampleRate);
        const float End = static_cast<float>(PendingSampleBase + Hop) / static_cast<float>(SampleRate);
        ProcessAnalysisFrame(Start, End, RMS, static_cast<float>(Crossings) / static_cast<float>(Hop), LowBandNorm, MidBandNorm, HighBandNorm, SpectralCentroidNorm, Periodicity);
        FOffgridAIStreamingAudioFeatureFrame& RichFrame = FeatureFrames.Last();
        RichFrame.RichLowBandNorm = RichLowBandNorm;
        RichFrame.RichMidBandNorm = RichMidBandNorm;
        RichFrame.RichHighBandNorm = RichHighBandNorm;
        RichFrame.RichSpectralCentroidNorm = RichCentroidNorm;
        RichFrame.RichSpectralRolloffNorm = RichRolloffNorm;
        RichFrame.RichSpectralFlatness = RichFlatness;
        RichFrame.RichSpectralFlux = RichFlux;
        RichFrame.RichPeriodicity = RichPeriodicity;
        PendingMonoSamples.RemoveAt(0, Hop, EAllowShrinking::No);
        PendingSampleBase += Hop;
    }
    ObservedAudioBufferEndSec = FMath::Max(ObservedAudioBufferEndSec, Frames > 0 ? (ChunkStartSample >= 0 ? static_cast<float>(ChunkStartSample + Frames) / static_cast<float>(SampleRate) : ObservedAudioBufferEndSec) : ObservedAudioBufferEndSec);
}


void FOffgridAIStreamingAudioFeatureExtractor::ProcessAnalysisFrame(
    float FrameStartSeconds,
    float FrameEndSeconds,
    float RMS,
    float ZCR,
    float LowBandNorm,
    float MidBandNorm,
    float HighBandNorm,
    float SpectralCentroidNorm,
    float Periodicity)
{
    SpeechPeakRMS = FMath::Max(SpeechPeakRMS * 0.995f, RMS);

    FOffgridAIStreamingAudioFeatureFrame Frame;
    Frame.AudioBufferStartSec = FrameStartSeconds;
    Frame.AudioBufferEndSec = FrameEndSeconds;
    Frame.AudioBufferCenterSec = 0.5f * (FrameStartSeconds + FrameEndSeconds);
    Frame.RMS = RMS;
    Frame.RMSNorm = FMath::Clamp(
        RMS / FMath::Max(SpeechPeakRMS, 0.0001f), 0.0f, 1.0f);
    Frame.DeltaRMS = FeatureFrames.Num() > 0
        ? RMS - FeatureFrames.Last().RMS
        : 0.0f;
    Frame.Flux = FMath::Max(Frame.DeltaRMS, 0.0f)
        / FMath::Max(SpeechPeakRMS, 0.0001f);
    Frame.ZCR = ZCR;
    Frame.LowBandNorm = LowBandNorm;
    Frame.MidBandNorm = MidBandNorm;
    Frame.HighBandNorm = HighBandNorm;
    Frame.SpectralCentroidNorm = SpectralCentroidNorm;
    Frame.Periodicity = Periodicity;
    FeatureFrames.Add(Frame);
    ObservedAudioBufferEndSec = FMath::Max(
        ObservedAudioBufferEndSec, FrameEndSeconds);
}

void FOffgridAIStreamingAudioFeatureExtractor::Finalize(
    float FinalObservedAudioBufferEndSec)
{
    if (FinalObservedAudioBufferEndSec >= 0.0f)
    {
        ObservedAudioBufferEndSec = FMath::Max(
            ObservedAudioBufferEndSec, FinalObservedAudioBufferEndSec);
    }
}
