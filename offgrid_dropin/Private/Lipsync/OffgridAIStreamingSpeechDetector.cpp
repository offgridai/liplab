#include "Lipsync/OffgridAIStreamingSpeechDetector.h"

namespace
{
constexpr float DetectorGapBridgeMaxSec = 0.180f;
constexpr float DetectorGapBridgeStrongQuietMaxSec = 0.200f;
constexpr float DetectorSoftBridgeWindowSec = 0.320f;
constexpr float DetectorEndpointBaseHangoverSec = 0.090f;
constexpr float DetectorEndpointStrongQuietHangoverSec = 0.060f;
constexpr float DetectorSoftRelativeCollapseThreshold = 0.18f;
constexpr float DetectorSoftRelativeCollapseReleaseThreshold = 0.28f;
constexpr float DetectorSoftRelativeCollapseEvidenceMax = 0.14f;
constexpr float DetectorSoftRelativeCollapseFluxMax = 0.030f;
constexpr float DetectorSoftRelativeCollapseHoldSec = 0.080f;
constexpr float DetectorRelativeCollapseThreshold = 0.05f;
constexpr float DetectorRelativeCollapseReleaseThreshold = 0.10f;
constexpr float DetectorRelativeCollapseHoldSec = 0.120f;
constexpr float DetectorHardRelativeCollapseThreshold = 0.02f;
constexpr float DetectorHardRelativeCollapseReleaseThreshold = 0.04f;
constexpr float DetectorHardRelativeCollapseHoldSec = 0.120f;
constexpr float DetectorStickyEndpointRelativeRMSMax = 0.08f;
constexpr float DetectorStickyEndpointFluxMax = 0.030f;
constexpr float DetectorShortGapBridgeMaxSec = 0.180f;
constexpr float DetectorShortGapBridgeReopenFluxMax = 0.060f;
constexpr float DetectorShortDeepValleySplitMinSec = 0.135f;
constexpr float DetectorShortDeepValleySplitMaxSec = 0.150f;
constexpr float DetectorShortDeepValleySplitMeanRMSMax = 0.020f;
constexpr float DetectorShortDeepValleySplitMinRMSMax = 0.0010f;
constexpr float DetectorShortDeepValleySplitPrevIslandMinSec = 1.500f;
constexpr float DetectorModerateGapBridgeMaxSec = 0.320f;
constexpr float DetectorModerateGapBridgeReopenFluxMax = 0.080f;
constexpr float DetectorModerateGapBridgeReopenEvidenceMax = 0.300f;
constexpr float DetectorShortIsolatedRestartSplitMinSec = 0.095f;
constexpr float DetectorShortIsolatedRestartSplitMaxSec = 0.125f;
constexpr float DetectorShortIsolatedRestartSplitMeanRMSMax = 0.050f;
constexpr float DetectorShortIsolatedRestartSplitMinRMSMax = 0.005f;
constexpr float DetectorShortIsolatedRestartSplitPrevIslandMaxSec = 0.600f;
constexpr float DetectorShortIsolatedRestartSplitReopenFluxMin = 0.300f;
constexpr float DetectorShortIsolatedRestartSplitReopenEvidenceMin = 0.380f;
constexpr float DetectorAmbiguousValleyBridgeMinSec = 0.170f;
constexpr float DetectorAmbiguousValleyBridgeMaxSec = 0.230f;
constexpr float DetectorAmbiguousValleyBridgeMeanRMSMax = 0.060f;
constexpr float DetectorAmbiguousValleyBridgeMinRMSMax = 0.004f;
constexpr float DetectorAmbiguousValleyBridgeReopenFluxMax = 0.190f;
constexpr float DetectorAmbiguousValleyBridgeReopenEvidenceMax = 0.560f;
constexpr float DetectorCollapsedRhetoricalSplitMinSec = 0.140f;
constexpr float DetectorCollapsedRhetoricalSplitMaxSec = 0.240f;
constexpr float DetectorCollapsedRhetoricalSplitMeanRMSMax = 0.022f;
constexpr float DetectorCollapsedRhetoricalSplitMinRMSMax = 0.0015f;
constexpr float DetectorCollapsedRhetoricalSplitPrevIslandMinSec = 0.700f;
constexpr float DetectorCollapsedRhetoricalSplitReopenFluxMin = 0.120f;
constexpr float DetectorCollapsedRhetoricalSplitReopenEvidenceMin = 0.350f;
constexpr float DetectorIsolatedPulseSplitMinSec = 0.095f;
constexpr float DetectorIsolatedPulseSplitMaxSec = 0.180f;
constexpr float DetectorIsolatedPulseSplitMeanRMSMax = 0.050f;
constexpr float DetectorIsolatedPulseSplitMinRMSMax = 0.006f;
constexpr float DetectorIsolatedPulseSplitReopenFluxMin = 0.160f;
constexpr float DetectorIsolatedPulseSplitPrevIslandMaxSec = 0.650f;
constexpr float DetectorShallowContinuityBridgeMinSec = 0.130f;
constexpr float DetectorShallowContinuityBridgeMaxSec = 0.160f;
constexpr float DetectorShallowContinuityBridgeMeanRMSMin = 0.020f;
constexpr float DetectorShallowContinuityBridgeReopenFluxMin = 0.030f;
constexpr float DetectorShallowContinuityBridgeReopenFluxMax = 0.100f;
constexpr float DetectorSharpRestartSplitMinSec = 0.090f;
constexpr float DetectorSharpRestartSplitMaxSec = 0.130f;
constexpr float DetectorSharpRestartSplitMeanRMSMax = 0.050f;
constexpr float DetectorSharpRestartSplitMinRMSMax = 0.010f;
constexpr float DetectorSharpRestartSplitReopenFluxMin = 0.180f;
constexpr float DetectorLeadingBlipMaxDurationSec = 0.160f;
constexpr float DetectorLeadingBlipReplacementMinGapSec = 0.120f;
constexpr float DetectorMicroIslandMaxDurationSec = 0.350f;
constexpr float DetectorMicroIslandPrevGapMinSec = 0.120f;
constexpr float DetectorMicroIslandNextGapMinSec = 0.180f;
constexpr float DetectorMicroIslandCombinedGapMinSec = 0.350f;
constexpr float DetectorSpeechBaselinePercentile = 0.60f;
constexpr int32 DetectorSpeechBaselineMinFrames = 12;

static float DetectorSpeechEvidence(float RMSNorm, float Flux, float Periodicity, float MidBandNorm, float HighBandNorm, float SpectralCentroidNorm)
{
    const float Voiced = Periodicity * 0.42f + RMSNorm * 0.24f + MidBandNorm * 0.10f;
    const float Unvoiced = Flux * 0.26f + HighBandNorm * 0.12f + SpectralCentroidNorm * 0.10f;
    return FMath::Clamp(Voiced + Unvoiced, 0.0f, 1.0f);
}

static bool DetectorStrongOnsetAnchor(float Evidence, float RMS, float NoiseFloorRMS, float Periodicity, float Flux)
{
    return Evidence >= 0.23f
        || (Periodicity >= 0.42f && RMS >= NoiseFloorRMS * 2.2f)
        || (Flux >= 0.18f && RMS >= NoiseFloorRMS * 2.8f);
}

static float DetectorBridgeWindowSec(FName EndReason)
{
    if (EndReason == FName(TEXT("strong_quiet_hangover"))
        || EndReason == FName(TEXT("finalize_at_provisional_end")))
    {
        return DetectorGapBridgeStrongQuietMaxSec;
    }
    return DetectorGapBridgeMaxSec;
}

static FOffgridAIStreamingPauseCue ClassifyPauseCue(
    bool bInSpeech,
    bool bEndpointCandidateActive,
    float GapAgeSec,
    bool bStrongQuiet,
    bool bLowEvidence,
    float Evidence,
    float RMSNorm,
    float Flux)
{
    FOffgridAIStreamingPauseCue Cue;
    Cue.bInSpeech = bInSpeech;
    Cue.bEndpointCandidateActive = bEndpointCandidateActive;
    Cue.GapAgeSec = FMath::Max(GapAgeSec, 0.0f);

    if (!bEndpointCandidateActive && bInSpeech)
    {
        Cue.Family = FName(TEXT("continuous_speech"));
        Cue.Confidence = FMath::Clamp(0.40f + Evidence * 0.45f + RMSNorm * 0.15f, 0.0f, 1.0f);
        return Cue;
    }

    const float QuietStrength = FMath::Clamp(
        (0.14f - Evidence) * 4.0f
        + (0.16f - Flux) * 1.4f
        + (0.22f - RMSNorm) * 1.2f
        + (bStrongQuiet ? 0.22f : 0.0f)
        + (bLowEvidence ? 0.10f : 0.0f),
        0.0f,
        1.0f);

    if (Cue.GapAgeSec >= 0.180f && QuietStrength >= 0.48f)
    {
        Cue.Family = FName(TEXT("hard_silence"));
        Cue.Confidence = FMath::Clamp(0.45f + QuietStrength * 0.55f, 0.0f, 1.0f);
        return Cue;
    }
    if (Cue.GapAgeSec >= 0.130f && QuietStrength >= 0.34f)
    {
        Cue.Family = FName(TEXT("phrase_gap"));
        Cue.Confidence = FMath::Clamp(0.32f + QuietStrength * 0.58f, 0.0f, 1.0f);
        return Cue;
    }
    if (Cue.GapAgeSec >= 0.065f && QuietStrength >= 0.20f)
    {
        Cue.Family = FName(TEXT("word_gap"));
        Cue.Confidence = FMath::Clamp(0.24f + QuietStrength * 0.52f, 0.0f, 1.0f);
        return Cue;
    }
    if (Cue.GapAgeSec >= 0.020f && QuietStrength >= 0.10f)
    {
        Cue.Family = FName(TEXT("micro_gap"));
        Cue.Confidence = FMath::Clamp(0.16f + QuietStrength * 0.48f, 0.0f, 1.0f);
        return Cue;
    }

    Cue.Family = FName(TEXT("continuous_speech"));
    Cue.Confidence = FMath::Clamp(0.20f + Evidence * 0.40f + RMSNorm * 0.10f, 0.0f, 1.0f);
    return Cue;
}

struct FDetectorGapDecision
{
    bool bBridge = false;
    FName DecisionClass = NAME_None;
};

static float SafeGapMean(float Sum, int32 Count)
{
    return Count > 0 ? Sum / static_cast<float>(Count) : 0.0f;
}

static float SpeechPercentileRMS(const TArray<float>& Samples, float Percentile)
{
    if (Samples.Num() <= 0)
    {
        return 0.0001f;
    }

    TArray<float> Sorted = Samples;
    Sorted.Sort([](float A, float B) { return A < B; });
    const float ClampedPercentile = FMath::Clamp(Percentile, 0.0f, 1.0f);
    const float Position = ClampedPercentile * static_cast<float>(FMath::Max(Sorted.Num() - 1, 0));
    const int32 LowerIndex = FMath::Clamp(static_cast<int32>(FMath::FloorToFloat(Position)), 0, Sorted.Num() - 1);
    const int32 UpperIndex = FMath::Clamp(
        LowerIndex + ((Position > static_cast<float>(LowerIndex)) ? 1 : 0),
        0,
        Sorted.Num() - 1);
    if (LowerIndex == UpperIndex)
    {
        return FMath::Max(Sorted[LowerIndex], 0.0001f);
    }

    const float Alpha = Position - static_cast<float>(LowerIndex);
    return FMath::Max(FMath::Lerp(Sorted[LowerIndex], Sorted[UpperIndex], Alpha), 0.0001f);
}

static void AccumulateGapFrame(
    FOffgridAIStreamingSpeechGapCandidate& Gap,
    float RMSNorm,
    float Evidence,
    float Flux,
    float Periodicity,
    float SpectralCentroidNorm,
    bool bStrongQuiet,
    bool bLowEvidence)
{
    Gap.GapFrameCount += 1;
    Gap.GapEvidenceSum += Evidence;
    Gap.GapRMSNormSum += RMSNorm;
    Gap.GapPeriodicitySum += Periodicity;
    Gap.GapFluxSum += Flux;
    Gap.GapCentroidSum += SpectralCentroidNorm;
    Gap.GapEvidenceMin = FMath::Min(Gap.GapEvidenceMin, Evidence);
    Gap.GapRMSNormMin = FMath::Min(Gap.GapRMSNormMin, RMSNorm);
    Gap.GapPeriodicityMax = FMath::Max(Gap.GapPeriodicityMax, Periodicity);
    Gap.GapFluxMax = FMath::Max(Gap.GapFluxMax, Flux);
    if (bLowEvidence)
    {
        Gap.LowEvidenceFrameCount += 1;
    }
    if (bStrongQuiet)
    {
        Gap.StrongQuietFrameCount += 1;
    }
    Gap.QuietEvidence = Gap.GapEvidenceMin;
    Gap.QuietRMSNorm = Gap.GapRMSNormMin;
}

static FDetectorGapDecision ClassifyGapDecision(
    const FOffgridAIStreamingSpeechIsland& PrevIsland,
    const FOffgridAIStreamingSpeechGapCandidate& Gap,
    float ReopenEvidence,
    float ReopenFlux,
    bool bStrongOnsetReopen)
{
    const float GapDurationSec = Gap.GapDurationSec;
    const float PrevIslandDurationSec = FMath::Max(PrevIsland.AudioBufferEndSec - PrevIsland.AudioBufferStartSec, 0.0f);
    if (GapDurationSec <= 0.055f)
    {
        return { true, FName(TEXT("micro_gap_bridge")) };
    }

    const float MeanEvidence = SafeGapMean(Gap.GapEvidenceSum, Gap.GapFrameCount);
    const float MeanRMSNorm = SafeGapMean(Gap.GapRMSNormSum, Gap.GapFrameCount);
    const float MeanPeriodicity = SafeGapMean(Gap.GapPeriodicitySum, Gap.GapFrameCount);
    const float LowEvidenceRatio = Gap.GapFrameCount > 0
        ? static_cast<float>(Gap.LowEvidenceFrameCount) / static_cast<float>(Gap.GapFrameCount)
        : 0.0f;
    const float StrongQuietRatio = Gap.GapFrameCount > 0
        ? static_cast<float>(Gap.StrongQuietFrameCount) / static_cast<float>(Gap.GapFrameCount)
        : 0.0f;
    const bool bStrongRestart = bStrongOnsetReopen || ReopenEvidence >= 0.23f || ReopenFlux >= 0.14f;

    const bool bCollapsedRhetoricalSplit =
        GapDurationSec >= DetectorCollapsedRhetoricalSplitMinSec
        && GapDurationSec <= DetectorCollapsedRhetoricalSplitMaxSec
        && MeanRMSNorm <= DetectorCollapsedRhetoricalSplitMeanRMSMax
        && Gap.GapRMSNormMin <= DetectorCollapsedRhetoricalSplitMinRMSMax
        && (PrevIslandDurationSec >= DetectorCollapsedRhetoricalSplitPrevIslandMinSec
            || ReopenFlux >= DetectorCollapsedRhetoricalSplitReopenFluxMin
            || ReopenEvidence >= DetectorCollapsedRhetoricalSplitReopenEvidenceMin)
        && (StrongQuietRatio >= 0.20f || LowEvidenceRatio >= 0.35f);
    if (bCollapsedRhetoricalSplit)
    {
        return { false, FName(TEXT("candidate_collapsed_rhetorical_split")) };
    }

    const bool bIsolatedPulseSplit =
        GapDurationSec >= DetectorIsolatedPulseSplitMinSec
        && GapDurationSec <= DetectorIsolatedPulseSplitMaxSec
        && MeanRMSNorm <= DetectorIsolatedPulseSplitMeanRMSMax
        && Gap.GapRMSNormMin <= DetectorIsolatedPulseSplitMinRMSMax
        && ReopenFlux >= DetectorIsolatedPulseSplitReopenFluxMin
        && PrevIslandDurationSec <= DetectorIsolatedPulseSplitPrevIslandMaxSec;
    if (bIsolatedPulseSplit)
    {
        return { false, FName(TEXT("candidate_isolated_pulse_split")) };
    }

    const bool bAmbiguousInternalValleyBridge =
        GapDurationSec >= DetectorAmbiguousValleyBridgeMinSec
        && GapDurationSec <= DetectorAmbiguousValleyBridgeMaxSec
        && MeanRMSNorm <= DetectorAmbiguousValleyBridgeMeanRMSMax
        && Gap.GapRMSNormMin <= DetectorAmbiguousValleyBridgeMinRMSMax
        && ReopenFlux <= DetectorAmbiguousValleyBridgeReopenFluxMax
        && ReopenEvidence <= DetectorAmbiguousValleyBridgeReopenEvidenceMax;
    if (bAmbiguousInternalValleyBridge)
    {
        return { true, FName(TEXT("candidate_ambiguous_valley_bridge")) };
    }

    const bool bShallowContinuityBridge =
        GapDurationSec >= DetectorShallowContinuityBridgeMinSec
        && GapDurationSec <= DetectorShallowContinuityBridgeMaxSec
        && MeanRMSNorm >= DetectorShallowContinuityBridgeMeanRMSMin
        && ReopenFlux >= DetectorShallowContinuityBridgeReopenFluxMin
        && ReopenFlux <= DetectorShallowContinuityBridgeReopenFluxMax;
    if (bShallowContinuityBridge)
    {
        return { true, FName(TEXT("candidate_shallow_continuity_bridge")) };
    }

    const bool bSharpRestartSplit =
        GapDurationSec >= DetectorSharpRestartSplitMinSec
        && GapDurationSec <= DetectorSharpRestartSplitMaxSec
        && MeanRMSNorm <= DetectorSharpRestartSplitMeanRMSMax
        && Gap.GapRMSNormMin <= DetectorSharpRestartSplitMinRMSMax
        && ReopenFlux >= DetectorSharpRestartSplitReopenFluxMin;
    if (bSharpRestartSplit)
    {
        return { false, FName(TEXT("candidate_sharp_restart_split")) };
    }

    const bool bShortDeepValleySplit =
        GapDurationSec >= DetectorShortDeepValleySplitMinSec
        && GapDurationSec <= DetectorShortDeepValleySplitMaxSec
        && MeanRMSNorm <= DetectorShortDeepValleySplitMeanRMSMax
        && Gap.GapRMSNormMin <= DetectorShortDeepValleySplitMinRMSMax
        && PrevIslandDurationSec >= DetectorShortDeepValleySplitPrevIslandMinSec
        && ReopenFlux <= DetectorShortGapBridgeReopenFluxMax;
    if (bShortDeepValleySplit)
    {
        return { false, FName(TEXT("candidate_short_deep_valley_split")) };
    }

    const bool bShortSoftBridge =
        GapDurationSec <= DetectorShortGapBridgeMaxSec
        && ReopenFlux <= DetectorShortGapBridgeReopenFluxMax
        && Gap.GapFluxMax <= 0.08f;
    if (bShortSoftBridge)
    {
        return { true, FName(TEXT("candidate_short_soft_bridge")) };
    }

    const bool bModerateSoftBridge =
        GapDurationSec <= DetectorModerateGapBridgeMaxSec
        && ReopenFlux <= DetectorModerateGapBridgeReopenFluxMax
        && ReopenEvidence <= DetectorModerateGapBridgeReopenEvidenceMax
        && Gap.GapFluxMax <= 0.06f
        && MeanRMSNorm <= 0.04f
        && !bStrongOnsetReopen;
    if (bModerateSoftBridge)
    {
        return { true, FName(TEXT("candidate_moderate_soft_bridge")) };
    }

    const bool bShortIsolatedRestartSplit =
        GapDurationSec >= DetectorShortIsolatedRestartSplitMinSec
        && GapDurationSec <= DetectorShortIsolatedRestartSplitMaxSec
        && MeanRMSNorm <= DetectorShortIsolatedRestartSplitMeanRMSMax
        && Gap.GapRMSNormMin <= DetectorShortIsolatedRestartSplitMinRMSMax
        && PrevIslandDurationSec <= DetectorShortIsolatedRestartSplitPrevIslandMaxSec
        && ReopenFlux >= DetectorShortIsolatedRestartSplitReopenFluxMin
        && ReopenEvidence >= DetectorShortIsolatedRestartSplitReopenEvidenceMin;
    if (bShortIsolatedRestartSplit)
    {
        return { false, FName(TEXT("candidate_short_isolated_restart_split")) };
    }

    int32 SplitScore = 0;
    if (GapDurationSec >= 0.115f) SplitScore += 2;
    else if (GapDurationSec >= 0.105f) SplitScore += 1;

    if (LowEvidenceRatio >= 0.78f) SplitScore += 2;
    else if (LowEvidenceRatio >= 0.58f) SplitScore += 1;

    if (StrongQuietRatio >= 0.34f) SplitScore += 1;
    if (MeanEvidence <= 0.095f) SplitScore += 1;
    if (Gap.GapEvidenceMin <= 0.070f) SplitScore += 1;
    if (MeanRMSNorm <= 0.18f) SplitScore += 1;
    if (Gap.GapRMSNormMin <= 0.11f) SplitScore += 1;
    if (bStrongRestart) SplitScore += 1;
    if (PrevIslandDurationSec >= 0.90f) SplitScore += 1;

    if (GapDurationSec < 0.100f) SplitScore -= 3;
    if (MeanPeriodicity >= 0.18f) SplitScore -= 1;
    if (Gap.GapFluxMax >= 0.18f && GapDurationSec < 0.090f) SplitScore -= 1;

    const bool bDeepQuietGap =
        GapDurationSec >= 0.105f
        && LowEvidenceRatio >= 0.78f
        && MeanEvidence <= 0.080f
        && MeanRMSNorm <= 0.15f
        && bStrongRestart;
    if (bDeepQuietGap)
    {
        return { false, FName(TEXT("candidate_deep_quiet_split")) };
    }

    const bool bDeepEnvelopeCollapseGap =
        GapDurationSec >= 0.180f
        && MeanRMSNorm <= 0.012f
        && Gap.GapRMSNormMin <= 0.005f
        && (LowEvidenceRatio >= 0.20f || StrongQuietRatio >= 0.15f);
    if (bDeepEnvelopeCollapseGap)
    {
        return { false, FName(TEXT("candidate_deep_envelope_split")) };
    }

    if (GapDurationSec >= 0.105f && SplitScore >= 8)
    {
        return { false, FName(TEXT("candidate_score_split")) };
    }

    return { true, FName(TEXT("candidate_score_bridge")) };
}
}

void FOffgridAIStreamingSpeechDetector::Reset()
{
    Islands.Reset();
    FeatureFrames.Reset();
    bInSpeech = false;
    bSpeechCandidateActive = false;
    SpeechCandidateStartSeconds = 0.0f;
    SpeechCandidateAccumSeconds = 0.0f;
    SpeechCandidatePeakEvidence = 0.0f;
    SilenceAccumSeconds = 0.0f;
    SilenceStartSeconds = 0.0f;
    bEndpointCandidateActive = false;
    EndpointCandidateStartSeconds = 0.0f;
    EndpointCandidateMinEvidence = 1.0f;
    EndpointCandidateFrameCount = 0;
    EndpointCandidateLowEvidenceFrameCount = 0;
    EndpointCandidateStrongQuietFrameCount = 0;
    EndpointCandidateEvidenceSum = 0.0f;
    EndpointCandidateRMSNormSum = 0.0f;
    EndpointCandidatePeriodicitySum = 0.0f;
    EndpointCandidateFluxSum = 0.0f;
    EndpointCandidateCentroidSum = 0.0f;
    EndpointCandidateMinRMSNorm = 1.0f;
    EndpointCandidateMaxPeriodicity = 0.0f;
    EndpointCandidateMaxFlux = 0.0f;
    ActiveIslandPeakRMS = 0.0001f;
    ActiveIslandSpeechRMSHistory.Reset();
    ActiveIslandSpeechSeconds = 0.0f;
    ActiveSoftCollapseAccumSeconds = 0.0f;
    ActiveSoftCollapseStartSeconds = 0.0f;
    ActiveLowEnergyAccumSeconds = 0.0f;
    ActiveLowEnergyStartSeconds = 0.0f;
    ActiveHardCollapseAccumSeconds = 0.0f;
    ActiveHardCollapseStartSeconds = 0.0f;
    bHasObservedFirstSpeechStart = false;
    FirstSpeechAudioBufferStartSec = 0.0f;
    ObservedAudioBufferEndSec = 0.0f;
    bPendingGapCandidateActive = false;
    PendingGapCandidate = FOffgridAIStreamingSpeechGapCandidate();
    PendingMonoSamples.Reset();
    PendingSampleBase = 0;
    ActiveSampleRate = 0;
    SpeechPeakRMS = 0.0001f;
    NoiseFloorRMS = 0.0001f;
    LatestPauseCue = FOffgridAIStreamingPauseCue();
}

void FOffgridAIStreamingSpeechDetector::SuppressRecentMicroIslandIfNeeded()
{
    if (Islands.Num() < 3)
    {
        return;
    }

    const int32 CurrentArrayIndex = Islands.Num() - 1;
    const int32 MiddleArrayIndex = CurrentArrayIndex - 1;
    const int32 PreviousArrayIndex = CurrentArrayIndex - 2;

    const FOffgridAIStreamingSpeechIsland& PreviousIsland = Islands[PreviousArrayIndex];
    const FOffgridAIStreamingSpeechIsland& MiddleIsland = Islands[MiddleArrayIndex];
    const FOffgridAIStreamingSpeechIsland& CurrentIsland = Islands[CurrentArrayIndex];

    const float MiddleDurationSec = FMath::Max(MiddleIsland.AudioBufferEndSec - MiddleIsland.AudioBufferStartSec, 0.0f);
    const float PreviousGapSec = FMath::Max(MiddleIsland.AudioBufferStartSec - PreviousIsland.AudioBufferEndSec, 0.0f);
    const float NextGapSec = FMath::Max(CurrentIsland.AudioBufferStartSec - MiddleIsland.AudioBufferEndSec, 0.0f);
    if (MiddleDurationSec > DetectorMicroIslandMaxDurationSec
        || PreviousGapSec < DetectorMicroIslandPrevGapMinSec
        || NextGapSec < DetectorMicroIslandNextGapMinSec
        || (PreviousGapSec + NextGapSec) < DetectorMicroIslandCombinedGapMinSec
        || MiddleIsland.ReopenCount > 3
        || MiddleIsland.EndReason != FName(TEXT("strong_quiet_hangover")))
    {
        return;
    }

    int32 LeftGapIndex = INDEX_NONE;
    int32 RightGapIndex = INDEX_NONE;
    for (int32 GapIndex = GapCandidates.Num() - 1; GapIndex >= 0; --GapIndex)
    {
        const FOffgridAIStreamingSpeechGapCandidate& Gap = GapCandidates[GapIndex];
        if (RightGapIndex == INDEX_NONE
            && !Gap.bBridged
            && Gap.PrevIslandIndex == MiddleIsland.IslandIndex
            && Gap.NextIslandIndex == CurrentIsland.IslandIndex)
        {
            RightGapIndex = GapIndex;
            continue;
        }
        if (LeftGapIndex == INDEX_NONE
            && !Gap.bBridged
            && Gap.PrevIslandIndex == PreviousIsland.IslandIndex
            && Gap.NextIslandIndex == MiddleIsland.IslandIndex)
        {
            LeftGapIndex = GapIndex;
        }
    }

    if (LeftGapIndex != INDEX_NONE)
    {
        FOffgridAIStreamingSpeechGapCandidate& LeftGap = GapCandidates[LeftGapIndex];
        LeftGap.GapEndSec = CurrentIsland.AudioBufferStartSec;
        LeftGap.GapDurationSec = FMath::Max(LeftGap.GapEndSec - LeftGap.GapStartSec, 0.0f);
        LeftGap.NextIslandIndex = CurrentIsland.IslandIndex;
        LeftGap.DecisionClass = FName(TEXT("candidate_micro_island_suppressed"));

        if (RightGapIndex != INDEX_NONE)
        {
            const FOffgridAIStreamingSpeechGapCandidate& RightGap = GapCandidates[RightGapIndex];
            LeftGap.ReopenEvidence = FMath::Max(LeftGap.ReopenEvidence, RightGap.ReopenEvidence);
            LeftGap.ReopenFlux = FMath::Max(LeftGap.ReopenFlux, RightGap.ReopenFlux);
            LeftGap.GapFrameCount += RightGap.GapFrameCount;
            LeftGap.LowEvidenceFrameCount += RightGap.LowEvidenceFrameCount;
            LeftGap.StrongQuietFrameCount += RightGap.StrongQuietFrameCount;
            LeftGap.GapEvidenceSum += RightGap.GapEvidenceSum;
            LeftGap.GapRMSNormSum += RightGap.GapRMSNormSum;
            LeftGap.GapPeriodicitySum += RightGap.GapPeriodicitySum;
            LeftGap.GapFluxSum += RightGap.GapFluxSum;
            LeftGap.GapCentroidSum += RightGap.GapCentroidSum;
            LeftGap.GapEvidenceMin = FMath::Min(LeftGap.GapEvidenceMin, RightGap.GapEvidenceMin);
            LeftGap.GapRMSNormMin = FMath::Min(LeftGap.GapRMSNormMin, RightGap.GapRMSNormMin);
            LeftGap.GapPeriodicityMax = FMath::Max(LeftGap.GapPeriodicityMax, RightGap.GapPeriodicityMax);
            LeftGap.GapFluxMax = FMath::Max(LeftGap.GapFluxMax, RightGap.GapFluxMax);
            LeftGap.QuietEvidence = FMath::Min(LeftGap.QuietEvidence, RightGap.QuietEvidence);
            LeftGap.QuietRMSNorm = FMath::Min(LeftGap.QuietRMSNorm, RightGap.QuietRMSNorm);
            LeftGap.bStrongQuietClose = LeftGap.bStrongQuietClose || RightGap.bStrongQuietClose;
        }
    }

    if (RightGapIndex != INDEX_NONE)
    {
        GapCandidates.RemoveAt(RightGapIndex);
        if (LeftGapIndex != INDEX_NONE && RightGapIndex < LeftGapIndex)
        {
            LeftGapIndex -= 1;
        }
    }

    const int32 RemovedIslandIndex = MiddleIsland.IslandIndex;
    Islands.RemoveAt(MiddleArrayIndex);
    for (int32 IslandIndex = 0; IslandIndex < Islands.Num(); ++IslandIndex)
    {
        Islands[IslandIndex].IslandIndex = IslandIndex;
    }
    for (FOffgridAIStreamingSpeechGapCandidate& Gap : GapCandidates)
    {
        if (Gap.PrevIslandIndex > RemovedIslandIndex)
        {
            Gap.PrevIslandIndex -= 1;
        }
        if (Gap.NextIslandIndex > RemovedIslandIndex)
        {
            Gap.NextIslandIndex -= 1;
        }
    }
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
    const bool bWasInSpeechAtFrameStart = bInSpeech;
    bool bFrameStartedIsland = false;
    bool bFrameClosedIsland = false;
    bool bFrameBridgedIsland = false;
    FName OccupancyDecision = NAME_None;

    const float Evidence = DetectorSpeechEvidence(Frame.RMSNorm, Frame.Flux, Frame.Periodicity, MidBandNorm, HighBandNorm, SpectralCentroidNorm);
    const bool bStrongOnsetAnchor = DetectorStrongOnsetAnchor(Evidence, RMS, NoiseFloorRMS, Periodicity, Frame.Flux);
    const bool bOpen = (RMS >= OpenThreshold && Evidence >= 0.16f) || bStrongOnsetAnchor;
    bool bKeepOpen = (RMS >= CloseThreshold && Evidence >= 0.12f)
        || Evidence >= 0.16f
        || (Periodicity >= 0.40f && RMS >= NoiseFloorRMS * 1.9f);
    const bool bStrongQuiet = Evidence < 0.065f
        && RMS < CloseThreshold * 0.82f
        && Periodicity < 0.24f
        && Frame.Flux < 0.045f;
    const bool bLowEvidence = bStrongQuiet
        || (Evidence < 0.08f && RMS < CloseThreshold * 0.90f);
    const float RecentSpeechBaselineRMS = ActiveIslandSpeechRMSHistory.Num() >= DetectorSpeechBaselineMinFrames
        ? SpeechPercentileRMS(ActiveIslandSpeechRMSHistory, DetectorSpeechBaselinePercentile)
        : FMath::Max(ActiveIslandPeakRMS, 0.0001f);
    const float ActiveIslandRelativeRMS = RMS / FMath::Max(RecentSpeechBaselineRMS, 0.0001f);
    const bool bSoftCollapseFrame = bInSpeech
        && ActiveIslandSpeechSeconds >= 0.120f
        && ActiveIslandRelativeRMS <= DetectorSoftRelativeCollapseThreshold
        && Evidence <= DetectorSoftRelativeCollapseEvidenceMax
        && Frame.Flux <= DetectorSoftRelativeCollapseFluxMax;
    const bool bSoftCollapseRelease =
        ActiveIslandRelativeRMS >= DetectorSoftRelativeCollapseReleaseThreshold
        || Evidence >= 0.18f
        || Frame.Flux >= 0.060f;
    const bool bRelativeCollapseFrame = bInSpeech
        && ActiveIslandSpeechSeconds >= 0.120f
        && ActiveIslandRelativeRMS <= DetectorRelativeCollapseThreshold;
    const bool bRelativeCollapseRelease = ActiveIslandRelativeRMS >= DetectorRelativeCollapseReleaseThreshold;
    const bool bHardRelativeCollapseFrame = bInSpeech
        && ActiveIslandSpeechSeconds >= 0.120f
        && ActiveIslandRelativeRMS <= DetectorHardRelativeCollapseThreshold;
    const bool bHardRelativeCollapseRelease = ActiveIslandRelativeRMS >= DetectorHardRelativeCollapseReleaseThreshold;
    const bool bStickyEndpointQuiet =
        bEndpointCandidateActive
        && ActiveIslandRelativeRMS <= DetectorStickyEndpointRelativeRMSMax
        && Frame.Flux <= DetectorStickyEndpointFluxMax;
    if (bStickyEndpointQuiet)
    {
        bKeepOpen = false;
    }

    if (bPendingGapCandidateActive && !bInSpeech && !bOpen)
    {
        AccumulateGapFrame(
            PendingGapCandidate,
            Frame.RMSNorm,
            Evidence,
            Frame.Flux,
            Periodicity,
            SpectralCentroidNorm,
            bStrongQuiet,
            bLowEvidence);
        PendingGapCandidate.GapEndSec = FrameEndSeconds;
        PendingGapCandidate.GapDurationSec = FMath::Max(PendingGapCandidate.GapEndSec - PendingGapCandidate.GapStartSec, 0.0f);
    }

    if (!bInSpeech)
    {
        if (bOpen)
        {
            if (!bSpeechCandidateActive)
            {
                bSpeechCandidateActive = true;
                SpeechCandidateStartSeconds = FrameStartSeconds;
                SpeechCandidateAccumSeconds = 0.0f;
                SpeechCandidatePeakEvidence = 0.0f;
            }
            SpeechCandidateAccumSeconds += FrameEndSeconds - FrameStartSeconds;
            SpeechCandidatePeakEvidence = FMath::Max(SpeechCandidatePeakEvidence, Evidence);
            if (SpeechCandidateAccumSeconds >= 0.035f && (bStrongOnsetAnchor || SpeechCandidatePeakEvidence >= 0.21f))
            {
                const bool bCanConsiderReopen = Islands.Num() > 0 && Islands.Last().bEnded && bPendingGapCandidateActive;
                const float PendingGapAgeSec = bCanConsiderReopen
                    ? FMath::Max(SpeechCandidateStartSeconds - Islands.Last().AudioBufferEndSec, 0.0f)
                    : FLT_MAX;
                const bool bCanConsiderSoftBridge = bCanConsiderReopen
                    && PendingGapAgeSec <= DetectorSoftBridgeWindowSec
                    && SpeechCandidatePeakEvidence <= DetectorModerateGapBridgeReopenEvidenceMax
                    && Frame.Flux <= DetectorModerateGapBridgeReopenFluxMax;
                if (Islands.Num() > 0
                    && Islands.Last().bEnded
                    && (PendingGapAgeSec <= DetectorBridgeWindowSec(Islands.Last().EndReason) || bCanConsiderSoftBridge))
                {
                    const float PendingMeanRMSNorm = SafeGapMean(PendingGapCandidate.GapRMSNormSum, PendingGapCandidate.GapFrameCount);
                    const bool bShortIsolatedRestartOverride =
                        PendingGapCandidate.GapDurationSec >= DetectorShortIsolatedRestartSplitMinSec
                        && PendingGapCandidate.GapDurationSec <= DetectorShortIsolatedRestartSplitMaxSec
                        && PendingMeanRMSNorm <= DetectorShortIsolatedRestartSplitMeanRMSMax
                        && PendingGapCandidate.GapRMSNormMin <= DetectorShortIsolatedRestartSplitMinRMSMax
                        && PendingGapCandidate.PrevIslandDurationSec <= DetectorShortIsolatedRestartSplitPrevIslandMaxSec
                        && Frame.Flux >= DetectorShortIsolatedRestartSplitReopenFluxMin
                        && SpeechCandidatePeakEvidence >= DetectorShortIsolatedRestartSplitReopenEvidenceMin;
                    const FDetectorGapDecision GapDecision = bShortIsolatedRestartOverride
                        ? FDetectorGapDecision{ false, FName(TEXT("candidate_short_isolated_restart_split")) }
                        : ClassifyGapDecision(
                            Islands.Last(),
                            PendingGapCandidate,
                            SpeechCandidatePeakEvidence,
                            Frame.Flux,
                            bStrongOnsetAnchor);
                    if (GapDecision.bBridge)
                    {
                        FOffgridAIStreamingSpeechIsland& Island = Islands.Last();
                        if (bPendingGapCandidateActive)
                        {
                            PendingGapCandidate.GapEndSec = SpeechCandidateStartSeconds;
                            PendingGapCandidate.GapDurationSec = FMath::Max(PendingGapCandidate.GapEndSec - PendingGapCandidate.GapStartSec, 0.0f);
                            PendingGapCandidate.ReopenEvidence = SpeechCandidatePeakEvidence;
                            PendingGapCandidate.ReopenFlux = Frame.Flux;
                            PendingGapCandidate.bStrongOnsetReopen = bStrongOnsetAnchor;
                            PendingGapCandidate.bBridged = true;
                            PendingGapCandidate.DecisionClass = GapDecision.DecisionClass;
                            PendingGapCandidate.NextIslandIndex = Island.IslandIndex;
                            PendingGapCandidate.GapIndex = GapCandidates.Num();
                            GapCandidates.Add(PendingGapCandidate);
                            bPendingGapCandidateActive = false;
                        }
                        Island.AudioBufferLastSpeechSec = FrameEndSeconds;
                        Island.AudioBufferEndSec = FrameEndSeconds;
                        Island.bEnded = false;
                        Island.EndReason = FName(TEXT("reopened"));
                        Island.ReopenCount += 1;
                        Island.ProvisionalEndSec = -1.0f;
                        Island.EndDecisionSec = -1.0f;
                        bFrameBridgedIsland = true;
                        OccupancyDecision = GapDecision.DecisionClass;
                    }
                    else
                    {
                        const float PrevIslandDurationSec = FMath::Max(
                            Islands.Last().AudioBufferEndSec - Islands.Last().AudioBufferStartSec,
                            0.0f);
                        const bool bReplaceLeadingBlip =
                            Islands.Num() == 1
                            && Islands.Last().AudioBufferStartSec <= 0.020f
                            && PrevIslandDurationSec <= DetectorLeadingBlipMaxDurationSec
                            && bPendingGapCandidateActive
                            && PendingGapCandidate.GapDurationSec >= DetectorLeadingBlipReplacementMinGapSec;
                        if (bReplaceLeadingBlip)
                        {
                            FOffgridAIStreamingSpeechIsland& Island = Islands.Last();
                            Island.AudioBufferStartSec = SpeechCandidateStartSeconds;
                            Island.AudioBufferLastSpeechSec = FrameEndSeconds;
                            Island.AudioBufferEndSec = FrameEndSeconds;
                            Island.bStarted = true;
                            Island.bEnded = false;
                            Island.ProvisionalEndSec = -1.0f;
                            Island.EndDecisionSec = -1.0f;
                            Island.EndReason = NAME_None;
                            Island.ReopenCount = 0;
                            bPendingGapCandidateActive = false;
                            bFrameStartedIsland = true;
                            OccupancyDecision = FName(TEXT("leading_blip_replaced"));
                        }
                        else
                        {
                            if (bPendingGapCandidateActive)
                            {
                                PendingGapCandidate.GapEndSec = SpeechCandidateStartSeconds;
                                PendingGapCandidate.GapDurationSec = FMath::Max(PendingGapCandidate.GapEndSec - PendingGapCandidate.GapStartSec, 0.0f);
                                PendingGapCandidate.ReopenEvidence = SpeechCandidatePeakEvidence;
                            PendingGapCandidate.ReopenFlux = Frame.Flux;
                            PendingGapCandidate.bStrongOnsetReopen = bStrongOnsetAnchor;
                            PendingGapCandidate.bBridged = false;
                            PendingGapCandidate.DecisionClass = GapDecision.DecisionClass;
                            PendingGapCandidate.NextIslandIndex = Islands.Num();
                            PendingGapCandidate.GapIndex = GapCandidates.Num();
                            GapCandidates.Add(PendingGapCandidate);
                            bPendingGapCandidateActive = false;
                        }

                            FOffgridAIStreamingSpeechIsland Island;
                            Island.IslandIndex = Islands.Num();
                            Island.AudioBufferStartSec = SpeechCandidateStartSeconds;
                            Island.AudioBufferLastSpeechSec = FrameEndSeconds;
                            Island.AudioBufferEndSec = FrameEndSeconds;
                            Island.bStarted = true;
                            Islands.Add(Island);
                            SuppressRecentMicroIslandIfNeeded();
                            bFrameStartedIsland = true;
                            OccupancyDecision = GapDecision.DecisionClass;
                        }
                    }
                }
                else
                {
                    const float PrevIslandDurationSec = (Islands.Num() > 0 && Islands.Last().bEnded)
                        ? FMath::Max(Islands.Last().AudioBufferEndSec - Islands.Last().AudioBufferStartSec, 0.0f)
                        : 0.0f;
                    const bool bReplaceLeadingBlip =
                        Islands.Num() == 1
                        && Islands.Last().bEnded
                        && Islands.Last().AudioBufferStartSec <= 0.020f
                        && PrevIslandDurationSec <= DetectorLeadingBlipMaxDurationSec
                        && bPendingGapCandidateActive
                        && PendingGapCandidate.GapDurationSec >= DetectorLeadingBlipReplacementMinGapSec;
                    if (bReplaceLeadingBlip)
                    {
                        FOffgridAIStreamingSpeechIsland& Island = Islands.Last();
                        Island.AudioBufferStartSec = SpeechCandidateStartSeconds;
                        Island.AudioBufferLastSpeechSec = FrameEndSeconds;
                        Island.AudioBufferEndSec = FrameEndSeconds;
                        Island.bStarted = true;
                        Island.bEnded = false;
                        Island.ProvisionalEndSec = -1.0f;
                        Island.EndDecisionSec = -1.0f;
                        Island.EndReason = NAME_None;
                        Island.ReopenCount = 0;
                        bPendingGapCandidateActive = false;
                        bFrameStartedIsland = true;
                        OccupancyDecision = FName(TEXT("leading_blip_replaced"));
                    }
                    else
                    {
                        if (bCanConsiderReopen && bPendingGapCandidateActive)
                        {
                            PendingGapCandidate.GapEndSec = SpeechCandidateStartSeconds;
                            PendingGapCandidate.GapDurationSec = FMath::Max(PendingGapCandidate.GapEndSec - PendingGapCandidate.GapStartSec, 0.0f);
                            PendingGapCandidate.ReopenEvidence = SpeechCandidatePeakEvidence;
                            PendingGapCandidate.ReopenFlux = Frame.Flux;
                            PendingGapCandidate.bStrongOnsetReopen = bStrongOnsetAnchor;
                            PendingGapCandidate.bBridged = false;
                            PendingGapCandidate.DecisionClass = FName(TEXT("bridge_window_expired_split"));
                            PendingGapCandidate.NextIslandIndex = Islands.Num();
                            PendingGapCandidate.GapIndex = GapCandidates.Num();
                            GapCandidates.Add(PendingGapCandidate);
                            bPendingGapCandidateActive = false;
                        }
                        FOffgridAIStreamingSpeechIsland Island;
                        Island.IslandIndex = Islands.Num();
                        Island.AudioBufferStartSec = SpeechCandidateStartSeconds;
                        Island.AudioBufferLastSpeechSec = FrameEndSeconds;
                        Island.AudioBufferEndSec = FrameEndSeconds;
                        Island.bStarted = true;
                        Islands.Add(Island);
                        SuppressRecentMicroIslandIfNeeded();
                        bFrameStartedIsland = true;
                        OccupancyDecision = bCanConsiderReopen ? FName(TEXT("bridge_window_expired_split")) : FName(TEXT("new_island_open"));
                    }
                }
                bInSpeech = true;
                bSpeechCandidateActive = false;
                SpeechCandidatePeakEvidence = 0.0f;
                ActiveIslandPeakRMS = RMS;
                ActiveIslandSpeechRMSHistory.Reset();
                ActiveIslandSpeechRMSHistory.Add(RMS);
                ActiveIslandSpeechSeconds = 0.0f;
                ActiveSoftCollapseAccumSeconds = 0.0f;
                ActiveSoftCollapseStartSeconds = 0.0f;
                ActiveLowEnergyAccumSeconds = 0.0f;
                ActiveLowEnergyStartSeconds = 0.0f;
                ActiveHardCollapseAccumSeconds = 0.0f;
                ActiveHardCollapseStartSeconds = 0.0f;
                SilenceAccumSeconds = 0.0f;
                bEndpointCandidateActive = false;
                EndpointCandidateStartSeconds = 0.0f;
                EndpointCandidateMinEvidence = 1.0f;
                EndpointCandidateFrameCount = 0;
                EndpointCandidateLowEvidenceFrameCount = 0;
                EndpointCandidateStrongQuietFrameCount = 0;
                EndpointCandidateEvidenceSum = 0.0f;
                EndpointCandidateRMSNormSum = 0.0f;
                EndpointCandidatePeriodicitySum = 0.0f;
                EndpointCandidateFluxSum = 0.0f;
                EndpointCandidateCentroidSum = 0.0f;
                EndpointCandidateMinRMSNorm = 1.0f;
                EndpointCandidateMaxPeriodicity = 0.0f;
                EndpointCandidateMaxFlux = 0.0f;
                if (!bHasObservedFirstSpeechStart)
                {
                    bHasObservedFirstSpeechStart = true;
                    FirstSpeechAudioBufferStartSec = SpeechCandidateStartSeconds;
                }
            }
        }
        else
        {
            if (bSpeechCandidateActive)
            {
                SpeechCandidateAccumSeconds = FMath::Max(0.0f, SpeechCandidateAccumSeconds - (FrameEndSeconds - FrameStartSeconds) * 0.5f);
                if (SpeechCandidateAccumSeconds <= 0.0f)
                {
                    bSpeechCandidateActive = false;
                    SpeechCandidatePeakEvidence = 0.0f;
                    OccupancyDecision = FName(TEXT("candidate_decay"));
                }
            }
        }
    }
    else
    {
        FOffgridAIStreamingSpeechIsland& Island = Islands.Last();
        Island.AudioBufferEndSec = FrameEndSeconds;
        ActiveIslandPeakRMS = FMath::Max(ActiveIslandPeakRMS, RMS);
        if (bKeepOpen && Evidence >= 0.16f && RMS >= CloseThreshold)
        {
            ActiveIslandSpeechRMSHistory.Add(RMS);
        }
        const float Dt = FrameEndSeconds - FrameStartSeconds;
        if (bSoftCollapseFrame)
        {
            if (ActiveSoftCollapseAccumSeconds <= 0.0f)
            {
                ActiveSoftCollapseStartSeconds = FrameStartSeconds;
            }
            ActiveSoftCollapseAccumSeconds += Dt;
        }
        else if (bSoftCollapseRelease)
        {
            ActiveSoftCollapseAccumSeconds = 0.0f;
            ActiveSoftCollapseStartSeconds = 0.0f;
        }
        else if (ActiveSoftCollapseAccumSeconds > 0.0f)
        {
            ActiveSoftCollapseAccumSeconds = FMath::Max(0.0f, ActiveSoftCollapseAccumSeconds - Dt * 0.5f);
            if (ActiveSoftCollapseAccumSeconds <= 0.0f)
            {
                ActiveSoftCollapseStartSeconds = 0.0f;
            }
        }
        if (bRelativeCollapseFrame)
        {
            if (ActiveLowEnergyAccumSeconds <= 0.0f)
            {
                ActiveLowEnergyStartSeconds = FrameStartSeconds;
            }
            ActiveLowEnergyAccumSeconds += Dt;
        }
        else if (bRelativeCollapseRelease)
        {
            ActiveLowEnergyAccumSeconds = 0.0f;
            ActiveLowEnergyStartSeconds = 0.0f;
        }
        else if (ActiveLowEnergyAccumSeconds > 0.0f)
        {
            ActiveLowEnergyAccumSeconds = FMath::Max(0.0f, ActiveLowEnergyAccumSeconds - Dt * 0.5f);
            if (ActiveLowEnergyAccumSeconds <= 0.0f)
            {
                ActiveLowEnergyStartSeconds = 0.0f;
            }
        }
        if (bHardRelativeCollapseFrame)
        {
            if (ActiveHardCollapseAccumSeconds <= 0.0f)
            {
                ActiveHardCollapseStartSeconds = FrameStartSeconds;
            }
            ActiveHardCollapseAccumSeconds += Dt;
        }
        else if (bHardRelativeCollapseRelease)
        {
            ActiveHardCollapseAccumSeconds = 0.0f;
            ActiveHardCollapseStartSeconds = 0.0f;
        }
        else if (ActiveHardCollapseAccumSeconds > 0.0f)
        {
            ActiveHardCollapseAccumSeconds = FMath::Max(0.0f, ActiveHardCollapseAccumSeconds - Dt * 0.5f);
            if (ActiveHardCollapseAccumSeconds <= 0.0f)
            {
                ActiveHardCollapseStartSeconds = 0.0f;
            }
        }

        const bool bSustainedSoftCollapse = ActiveSoftCollapseAccumSeconds >= DetectorSoftRelativeCollapseHoldSec;
        const bool bSustainedRelativeCollapse = ActiveLowEnergyAccumSeconds >= DetectorRelativeCollapseHoldSec;
        const bool bSustainedHardRelativeCollapse = ActiveHardCollapseAccumSeconds >= DetectorHardRelativeCollapseHoldSec;
        if (bHardRelativeCollapseFrame)
        {
            bKeepOpen = false;
        }
        if (bKeepOpen)
        {
            Island.AudioBufferLastSpeechSec = FrameEndSeconds;
            ActiveIslandSpeechSeconds += FrameEndSeconds - FrameStartSeconds;
            if (bEndpointCandidateActive)
            {
                EndpointCandidateMinEvidence = FMath::Min(EndpointCandidateMinEvidence, Evidence);
                Island.ReopenCount += 1;
            }
            SilenceAccumSeconds = 0.0f;
            if (bSustainedRelativeCollapse || bSustainedSoftCollapse)
            {
                if (!bEndpointCandidateActive)
                {
                    bEndpointCandidateActive = true;
                    EndpointCandidateStartSeconds = bSustainedRelativeCollapse && ActiveLowEnergyStartSeconds > 0.0f
                        ? ActiveLowEnergyStartSeconds
                        : (bSustainedSoftCollapse && ActiveSoftCollapseStartSeconds > 0.0f
                            ? ActiveSoftCollapseStartSeconds
                            : FrameStartSeconds);
                    EndpointCandidateMinEvidence = Evidence;
                    Island.ProvisionalEndSec = EndpointCandidateStartSeconds;
                    OccupancyDecision = bSustainedRelativeCollapse
                        ? FName(TEXT("relative_collapse_candidate_start"))
                        : FName(TEXT("soft_collapse_candidate_start"));
                }
                bKeepOpen = false;
            }
            else
            {
                bEndpointCandidateActive = false;
                EndpointCandidateStartSeconds = 0.0f;
                EndpointCandidateMinEvidence = 1.0f;
                Island.ProvisionalEndSec = -1.0f;
                OccupancyDecision = FName(TEXT("keep_open"));
            }
        }
        if (!bKeepOpen)
        {
            if (SilenceAccumSeconds <= 0.0f)
            {
                if (bSustainedHardRelativeCollapse && ActiveHardCollapseStartSeconds > 0.0f)
                {
                    SilenceStartSeconds = ActiveHardCollapseStartSeconds;
                }
                else if (bSustainedRelativeCollapse && ActiveLowEnergyStartSeconds > 0.0f)
                {
                    SilenceStartSeconds = ActiveLowEnergyStartSeconds;
                }
                else if (bSustainedSoftCollapse && ActiveSoftCollapseStartSeconds > 0.0f)
                {
                    SilenceStartSeconds = ActiveSoftCollapseStartSeconds;
                }
                else
                {
                    SilenceStartSeconds = FrameStartSeconds;
                }
                bEndpointCandidateActive = true;
                EndpointCandidateStartSeconds = SilenceStartSeconds;
                EndpointCandidateMinEvidence = Evidence;
                Island.ProvisionalEndSec = SilenceStartSeconds;
                if (bSustainedHardRelativeCollapse)
                {
                    OccupancyDecision = FName(TEXT("hard_relative_collapse_candidate_start"));
                }
                else if (bSustainedRelativeCollapse)
                {
                    OccupancyDecision = FName(TEXT("relative_collapse_candidate_start"));
                }
                else if (bSustainedSoftCollapse)
                {
                    OccupancyDecision = FName(TEXT("soft_collapse_candidate_start"));
                }
                else
                {
                    OccupancyDecision = FName(TEXT("endpoint_candidate_start"));
                }
            }
            if (bEndpointCandidateActive)
            {
                EndpointCandidateMinEvidence = FMath::Min(EndpointCandidateMinEvidence, Evidence);
                EndpointCandidateFrameCount += 1;
                EndpointCandidateEvidenceSum += Evidence;
                EndpointCandidateRMSNormSum += Frame.RMSNorm;
                EndpointCandidatePeriodicitySum += Periodicity;
                EndpointCandidateFluxSum += Frame.Flux;
                EndpointCandidateCentroidSum += SpectralCentroidNorm;
                EndpointCandidateMinRMSNorm = FMath::Min(EndpointCandidateMinRMSNorm, Frame.RMSNorm);
                EndpointCandidateMaxPeriodicity = FMath::Max(EndpointCandidateMaxPeriodicity, Periodicity);
                EndpointCandidateMaxFlux = FMath::Max(EndpointCandidateMaxFlux, Frame.Flux);
                if (bLowEvidence)
                {
                    EndpointCandidateLowEvidenceFrameCount += 1;
                }
                if (bStrongQuiet)
                {
                    EndpointCandidateStrongQuietFrameCount += 1;
                }
            }
            SilenceAccumSeconds += bSustainedHardRelativeCollapse ? Dt : (bLowEvidence ? Dt : Dt * 0.50f);
            const float Hangover = bSustainedHardRelativeCollapse
                ? DetectorHardRelativeCollapseHoldSec
                : (bSustainedRelativeCollapse
                    ? DetectorRelativeCollapseHoldSec
                    : (bSustainedSoftCollapse
                        ? DetectorSoftRelativeCollapseHoldSec
                        : (bStrongQuiet ? DetectorEndpointStrongQuietHangoverSec : DetectorEndpointBaseHangoverSec)));
            if (SilenceAccumSeconds >= Hangover)
            {
                Island.AudioBufferEndSec = EndpointCandidateStartSeconds;
                Island.EndDecisionSec = FrameEndSeconds;
                if (bSustainedHardRelativeCollapse)
                {
                    Island.EndReason = FName(TEXT("hard_relative_collapse_hangover"));
                }
                else if (bSustainedRelativeCollapse)
                {
                    Island.EndReason = FName(TEXT("relative_collapse_hangover"));
                }
                else if (bSustainedSoftCollapse)
                {
                    Island.EndReason = FName(TEXT("soft_collapse_hangover"));
                }
                else
                {
                    Island.EndReason = bStrongQuiet ? FName(TEXT("strong_quiet_hangover")) : FName(TEXT("weak_evidence_hangover"));
                }
                PendingGapCandidate = FOffgridAIStreamingSpeechGapCandidate();
                PendingGapCandidate.PrevIslandIndex = Island.IslandIndex;
                PendingGapCandidate.GapStartSec = EndpointCandidateStartSeconds;
                PendingGapCandidate.PrevIslandDurationSec = FMath::Max(Island.AudioBufferEndSec - Island.AudioBufferStartSec, 0.0f);
                PendingGapCandidate.GapEndSec = FrameEndSeconds;
                PendingGapCandidate.GapDurationSec = FMath::Max(PendingGapCandidate.GapEndSec - PendingGapCandidate.GapStartSec, 0.0f);
                PendingGapCandidate.GapFrameCount = EndpointCandidateFrameCount;
                PendingGapCandidate.LowEvidenceFrameCount = EndpointCandidateLowEvidenceFrameCount;
                PendingGapCandidate.StrongQuietFrameCount = EndpointCandidateStrongQuietFrameCount;
                PendingGapCandidate.GapEvidenceSum = EndpointCandidateEvidenceSum;
                PendingGapCandidate.GapRMSNormSum = EndpointCandidateRMSNormSum;
                PendingGapCandidate.GapPeriodicitySum = EndpointCandidatePeriodicitySum;
                PendingGapCandidate.GapFluxSum = EndpointCandidateFluxSum;
                PendingGapCandidate.GapCentroidSum = EndpointCandidateCentroidSum;
                PendingGapCandidate.GapEvidenceMin = EndpointCandidateMinEvidence;
                PendingGapCandidate.GapRMSNormMin = EndpointCandidateMinRMSNorm;
                PendingGapCandidate.GapPeriodicityMax = EndpointCandidateMaxPeriodicity;
                PendingGapCandidate.GapFluxMax = EndpointCandidateMaxFlux;
                PendingGapCandidate.QuietEvidence = EndpointCandidateMinEvidence;
                PendingGapCandidate.QuietRMSNorm = EndpointCandidateMinRMSNorm;
                PendingGapCandidate.bStrongQuietClose = bStrongQuiet;
                PendingGapCandidate.CloseReason = Island.EndReason;
                bPendingGapCandidateActive = true;
                Island.bEnded = true;
                bFrameClosedIsland = true;
                OccupancyDecision = Island.EndReason;
                bInSpeech = false;
                SilenceAccumSeconds = 0.0f;
                ActiveIslandSpeechRMSHistory.Reset();
                ActiveSoftCollapseAccumSeconds = 0.0f;
                ActiveSoftCollapseStartSeconds = 0.0f;
                ActiveLowEnergyAccumSeconds = 0.0f;
                ActiveLowEnergyStartSeconds = 0.0f;
                ActiveHardCollapseAccumSeconds = 0.0f;
                ActiveHardCollapseStartSeconds = 0.0f;
                bEndpointCandidateActive = false;
                EndpointCandidateStartSeconds = 0.0f;
                EndpointCandidateMinEvidence = 1.0f;
                EndpointCandidateFrameCount = 0;
                EndpointCandidateLowEvidenceFrameCount = 0;
                EndpointCandidateStrongQuietFrameCount = 0;
                EndpointCandidateEvidenceSum = 0.0f;
                EndpointCandidateRMSNormSum = 0.0f;
                EndpointCandidatePeriodicitySum = 0.0f;
                EndpointCandidateFluxSum = 0.0f;
                EndpointCandidateCentroidSum = 0.0f;
                EndpointCandidateMinRMSNorm = 1.0f;
                EndpointCandidateMaxPeriodicity = 0.0f;
                EndpointCandidateMaxFlux = 0.0f;
            }
        }
    }

    if (OccupancyDecision.IsNone())
    {
        if (!bWasInSpeechAtFrameStart && !bOpen)
        {
            OccupancyDecision = bSpeechCandidateActive ? FName(TEXT("candidate_accumulating")) : FName(TEXT("closed_frame"));
        }
        else if (bWasInSpeechAtFrameStart && !bKeepOpen)
        {
            OccupancyDecision = FName(TEXT("endpoint_candidate_accumulating"));
        }
        else
        {
            OccupancyDecision = FName(TEXT("no_state_change"));
        }
    }

    Frame.SpeechEvidence = Evidence;
    Frame.OpenThreshold = OpenThreshold;
    Frame.CloseThreshold = CloseThreshold;
    Frame.SilenceAccumSec = SilenceAccumSeconds;
    Frame.EndpointCandidateStartSec = bEndpointCandidateActive ? EndpointCandidateStartSeconds : -1.0f;
    Frame.ActiveIslandStartSec = Islands.Num() > 0 ? Islands.Last().AudioBufferStartSec : -1.0f;
    Frame.ActiveIslandEndSec = Islands.Num() > 0 ? Islands.Last().AudioBufferEndSec : -1.0f;
    Frame.bInSpeechBeforeFrame = bWasInSpeechAtFrameStart;
    Frame.bInSpeechAfterFrame = bInSpeech;
    Frame.bOpenCandidate = bOpen;
    Frame.bKeepOpen = bKeepOpen;
    Frame.bStrongOnsetAnchor = bStrongOnsetAnchor;
    Frame.bStrongQuiet = bStrongQuiet;
    Frame.bLowEvidence = bLowEvidence;
    Frame.bEndpointCandidateActive = bEndpointCandidateActive;
    Frame.bFrameStartedIsland = bFrameStartedIsland;
    Frame.bFrameClosedIsland = bFrameClosedIsland;
    Frame.bFrameBridgedIsland = bFrameBridgedIsland;
    Frame.OccupancyDecision = OccupancyDecision;

    float PauseGapAgeSec = 0.0f;
    if (bEndpointCandidateActive)
    {
        PauseGapAgeSec = FMath::Max(FrameEndSeconds - EndpointCandidateStartSeconds, 0.0f);
    }
    else if (!bInSpeech && Islands.Num() > 0 && Islands.Last().bEnded)
    {
        PauseGapAgeSec = FMath::Max(FrameEndSeconds - Islands.Last().AudioBufferEndSec, 0.0f);
    }
    LatestPauseCue = ClassifyPauseCue(
        bInSpeech,
        bEndpointCandidateActive,
        PauseGapAgeSec,
        bStrongQuiet,
        bLowEvidence,
        Evidence,
        Frame.RMSNorm,
        Frame.Flux);
    Frame.PauseFamily = LatestPauseCue.Family;
    Frame.PauseFamilyConfidence = LatestPauseCue.Confidence;
    Frame.PauseGapAgeSec = LatestPauseCue.GapAgeSec;

    FeatureFrames.Add(Frame);

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
        FOffgridAIStreamingSpeechIsland& Island = Islands.Last();
        Island.bEnded = true;

        // Finalization is not fresh acoustic speech. If an endpoint candidate is
        // already active, close at the first quiet frame. Otherwise close at the
        // last frame that actually met the keep-open speech criteria. This keeps
        // trailing buffering/final-drain mechanics out of detector speech-region
        // scoring.
        if (bEndpointCandidateActive && EndpointCandidateStartSeconds > Island.AudioBufferStartSec)
        {
            Island.AudioBufferEndSec = EndpointCandidateStartSeconds;
            Island.ProvisionalEndSec = EndpointCandidateStartSeconds;
            Island.EndDecisionSec = FinalObservedAudioBufferEndSec >= 0.0f ? FinalObservedAudioBufferEndSec : ObservedAudioBufferEndSec;
            Island.EndReason = FName(TEXT("finalize_at_provisional_end"));
            PendingGapCandidate = FOffgridAIStreamingSpeechGapCandidate();
            PendingGapCandidate.PrevIslandIndex = Island.IslandIndex;
            PendingGapCandidate.GapStartSec = EndpointCandidateStartSeconds;
            PendingGapCandidate.PrevIslandDurationSec = FMath::Max(Island.AudioBufferEndSec - Island.AudioBufferStartSec, 0.0f);
            PendingGapCandidate.GapEndSec = FinalObservedAudioBufferEndSec >= 0.0f ? FinalObservedAudioBufferEndSec : ObservedAudioBufferEndSec;
            PendingGapCandidate.GapDurationSec = FMath::Max(PendingGapCandidate.GapEndSec - PendingGapCandidate.GapStartSec, 0.0f);
            PendingGapCandidate.GapFrameCount = EndpointCandidateFrameCount;
            PendingGapCandidate.LowEvidenceFrameCount = EndpointCandidateLowEvidenceFrameCount;
            PendingGapCandidate.StrongQuietFrameCount = EndpointCandidateStrongQuietFrameCount;
            PendingGapCandidate.GapEvidenceSum = EndpointCandidateEvidenceSum;
            PendingGapCandidate.GapRMSNormSum = EndpointCandidateRMSNormSum;
            PendingGapCandidate.GapPeriodicitySum = EndpointCandidatePeriodicitySum;
            PendingGapCandidate.GapFluxSum = EndpointCandidateFluxSum;
            PendingGapCandidate.GapCentroidSum = EndpointCandidateCentroidSum;
            PendingGapCandidate.GapEvidenceMin = EndpointCandidateMinEvidence;
            PendingGapCandidate.GapRMSNormMin = EndpointCandidateMinRMSNorm;
            PendingGapCandidate.GapPeriodicityMax = EndpointCandidateMaxPeriodicity;
            PendingGapCandidate.GapFluxMax = EndpointCandidateMaxFlux;
            PendingGapCandidate.QuietEvidence = EndpointCandidateMinEvidence;
            PendingGapCandidate.QuietRMSNorm = EndpointCandidateMinRMSNorm;
            PendingGapCandidate.bStrongQuietClose = true;
            PendingGapCandidate.CloseReason = Island.EndReason;
            bPendingGapCandidateActive = true;
        }
        else if (Island.AudioBufferLastSpeechSec > Island.AudioBufferStartSec)
        {
            Island.AudioBufferEndSec = Island.AudioBufferLastSpeechSec;
            Island.EndDecisionSec = FinalObservedAudioBufferEndSec >= 0.0f ? FinalObservedAudioBufferEndSec : ObservedAudioBufferEndSec;
            Island.EndReason = FName(TEXT("finalize_at_last_speech"));
        }
        else
        {
            Island.AudioBufferEndSec = FinalObservedAudioBufferEndSec >= 0.0f ? FinalObservedAudioBufferEndSec : ObservedAudioBufferEndSec;
            Island.EndDecisionSec = Island.AudioBufferEndSec;
            Island.EndReason = FName(TEXT("finalize_no_candidate"));
        }
    }
    bInSpeech = false;
    bEndpointCandidateActive = false;
    EndpointCandidateStartSeconds = 0.0f;
    EndpointCandidateMinEvidence = 1.0f;
    SilenceAccumSeconds = 0.0f;
    ActiveSoftCollapseAccumSeconds = 0.0f;
    ActiveSoftCollapseStartSeconds = 0.0f;
    ActiveLowEnergyAccumSeconds = 0.0f;
    ActiveLowEnergyStartSeconds = 0.0f;
    ActiveHardCollapseAccumSeconds = 0.0f;
    ActiveHardCollapseStartSeconds = 0.0f;
    bPendingGapCandidateActive = false;
}
