#include "Lipsync/OffgridAIStreamingSpeechDetector.h"

void FOffgridAIStreamingSpeechDetector::Reset()
{
    Islands.Reset();
    FeatureFrames.Reset();
    bInSpeech = false;
    bSpeechCandidateActive = false;
    SpeechCandidateStartSeconds = 0.0f;
    SpeechCandidateAccumSeconds = 0.0f;
    SilenceAccumSeconds = 0.0f;
    SilenceStartSeconds = 0.0f;
    ActiveIslandPeakRMS = 0.0001f;
    ActiveIslandSpeechSeconds = 0.0f;
    ActiveLowEnergyAccumSeconds = 0.0f;
    ActiveLowEnergyStartSeconds = 0.0f;
    bHasObservedFirstSpeechStart = false;
    FirstSpeechAudioBufferStartSec = 0.0f;
    ObservedAudioBufferEndSec = 0.0f;
    PendingMonoSamples.Reset();
    PendingSampleBase = 0;
    ActiveSampleRate = 0;
    SpeechPeakRMS = 0.0001f;
    NoiseFloorRMS = 0.0001f;
}

void FOffgridAIStreamingSpeechDetector::AppendPCM16(const TArray<uint8>& PCMChunk, int32 BytesToUse, int32 SampleRate, int32 NumChannels, int64 ChunkStartSample)
{
    if (BytesToUse <= 0 || SampleRate <= 0 || NumChannels <= 0)
    {
        return;
    }
    if (ActiveSampleRate != SampleRate)
    {
        PendingMonoSamples.Reset();
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

        const float Start = static_cast<float>(PendingSampleBase) / static_cast<float>(SampleRate);
        const float End = static_cast<float>(PendingSampleBase + Hop) / static_cast<float>(SampleRate);
        ProcessAnalysisFrame(Start, End, RMS, static_cast<float>(Crossings) / static_cast<float>(Hop), LowBandNorm, MidBandNorm, HighBandNorm, SpectralCentroidNorm, Periodicity);
        PendingMonoSamples.RemoveAt(0, Hop, EAllowShrinking::No);
        PendingSampleBase += Hop;
    }
    ObservedAudioBufferEndSec = FMath::Max(ObservedAudioBufferEndSec, Frames > 0 ? (ChunkStartSample >= 0 ? static_cast<float>(ChunkStartSample + Frames) / static_cast<float>(SampleRate) : ObservedAudioBufferEndSec) : ObservedAudioBufferEndSec);
}

void FOffgridAIStreamingSpeechDetector::ProcessAnalysisFrame(float FrameStartSeconds, float FrameEndSeconds, float RMS, float ZCR, float LowBandNorm, float MidBandNorm, float HighBandNorm, float SpectralCentroidNorm, float Periodicity)
{
    SpeechPeakRMS = FMath::Max(SpeechPeakRMS * 0.995f, RMS);
    if (!bInSpeech)
    {
        NoiseFloorRMS = FMath::Lerp(NoiseFloorRMS, RMS, 0.02f);
    }
    const float OpenThreshold = FMath::Max(0.0065f, NoiseFloorRMS * 3.2f);
    const float CloseThreshold = FMath::Max(0.0035f, NoiseFloorRMS * 1.9f);

    FOffgridAIStreamingAudioFeatureFrame Frame;
    Frame.AudioBufferStartSec = FrameStartSeconds;
    Frame.AudioBufferEndSec = FrameEndSeconds;
    Frame.AudioBufferCenterSec = (FrameStartSeconds + FrameEndSeconds) * 0.5f;
    Frame.RMS = RMS;
    Frame.RMSNorm = FMath::Clamp(RMS / FMath::Max(SpeechPeakRMS, 0.0001f), 0.0f, 1.0f);
    Frame.DeltaRMS = FeatureFrames.Num() > 0 ? RMS - FeatureFrames.Last().RMS : 0.0f;
    Frame.Flux = FMath::Max(Frame.DeltaRMS, 0.0f) / FMath::Max(SpeechPeakRMS, 0.0001f);
    Frame.ZCR = ZCR;
    Frame.LowBandNorm = LowBandNorm;
    Frame.MidBandNorm = MidBandNorm;
    Frame.HighBandNorm = HighBandNorm;
    Frame.SpectralCentroidNorm = SpectralCentroidNorm;
    Frame.Periodicity = Periodicity;
    FeatureFrames.Add(Frame);

    const bool bOpen = RMS >= OpenThreshold;
    const bool bKeepOpen = RMS >= CloseThreshold;
    if (!bInSpeech)
    {
        if (bOpen)
        {
            if (!bSpeechCandidateActive)
            {
                bSpeechCandidateActive = true;
                SpeechCandidateStartSeconds = FrameStartSeconds;
                SpeechCandidateAccumSeconds = 0.0f;
            }
            SpeechCandidateAccumSeconds += FrameEndSeconds - FrameStartSeconds;
            if (SpeechCandidateAccumSeconds >= 0.025f)
            {
                FOffgridAIStreamingSpeechIsland Island;
                Island.IslandIndex = Islands.Num();
                Island.AudioBufferStartSec = SpeechCandidateStartSeconds;
                Island.AudioBufferLastSpeechSec = FrameEndSeconds;
                Island.AudioBufferEndSec = FrameEndSeconds;
                Island.bStarted = true;
                Islands.Add(Island);
                bInSpeech = true;
                bSpeechCandidateActive = false;
                ActiveIslandPeakRMS = RMS;
                ActiveIslandSpeechSeconds = 0.0f;
                SilenceAccumSeconds = 0.0f;
                if (!bHasObservedFirstSpeechStart)
                {
                    bHasObservedFirstSpeechStart = true;
                    FirstSpeechAudioBufferStartSec = SpeechCandidateStartSeconds;
                }
            }
        }
        else
        {
            bSpeechCandidateActive = false;
            SpeechCandidateAccumSeconds = 0.0f;
        }
    }
    else
    {
        FOffgridAIStreamingSpeechIsland& Island = Islands.Last();
        Island.AudioBufferEndSec = FrameEndSeconds;
        ActiveIslandPeakRMS = FMath::Max(ActiveIslandPeakRMS, RMS);
        if (bKeepOpen)
        {
            Island.AudioBufferLastSpeechSec = FrameEndSeconds;
            ActiveIslandSpeechSeconds += FrameEndSeconds - FrameStartSeconds;
            SilenceAccumSeconds = 0.0f;
        }
        else
        {
            if (SilenceAccumSeconds <= 0.0f) SilenceStartSeconds = FrameStartSeconds;
            SilenceAccumSeconds += FrameEndSeconds - FrameStartSeconds;
            if (SilenceAccumSeconds >= 0.090f)
            {
                Island.AudioBufferEndSec = SilenceStartSeconds;
                Island.bEnded = true;
                bInSpeech = false;
                SilenceAccumSeconds = 0.0f;
            }
        }
    }

    ObservedAudioBufferEndSec = FMath::Max(ObservedAudioBufferEndSec, FrameEndSeconds);
    RefreshLocalFeatureFlags();
}

void FOffgridAIStreamingSpeechDetector::RefreshLocalFeatureFlags()
{
    if (FeatureFrames.Num() < 3) return;
    const int32 I = FeatureFrames.Num() - 2;
    FOffgridAIStreamingAudioFeatureFrame& F = FeatureFrames[I];
    const FOffgridAIStreamingAudioFeatureFrame& A = FeatureFrames[I - 1];
    const FOffgridAIStreamingAudioFeatureFrame& B = FeatureFrames[I + 1];
    F.bLocalRMSPeak = F.RMS > A.RMS && F.RMS >= B.RMS;
    F.bLocalRMSValley = F.RMS < A.RMS && F.RMS <= B.RMS;
    F.bLocalFluxPeak = F.Flux > A.Flux && F.Flux >= B.Flux;
}

void FOffgridAIStreamingSpeechDetector::Finalize(float FinalObservedAudioBufferEndSec)
{
    if (Islands.Num() > 0 && bInSpeech)
    {
        Islands.Last().bEnded = true;
        Islands.Last().AudioBufferEndSec = FinalObservedAudioBufferEndSec >= 0.0f ? FinalObservedAudioBufferEndSec : ObservedAudioBufferEndSec;
    }
    bInSpeech = false;
}
