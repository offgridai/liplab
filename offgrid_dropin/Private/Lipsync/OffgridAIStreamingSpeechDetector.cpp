#include "Lipsync/OffgridAIStreamingSpeechDetector.h"

#include <cmath>

#include "OffgridAIStreamingRegionModel.inl"

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
constexpr float DetectorAdvisorySoftLullMinSec = 0.030f;
constexpr float DetectorAdvisorySoftLullMergeGapSec = 0.060f;
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
constexpr float DetectorClosureLikeBridgeMinSec = 0.085f;
constexpr float DetectorClosureLikeBridgeMaxSec = 0.135f;
constexpr float DetectorClosureLikeBridgeMeanRMSMax = 0.055f;
constexpr float DetectorClosureLikeBridgeMinRMSMax = 0.010f;
constexpr float DetectorClosureLikeBridgeLowEvidenceRatioMax = 0.82f;
constexpr float DetectorClosureLikeBridgeStrongQuietRatioMax = 0.38f;
constexpr float DetectorClosureLikeBridgeReopenFluxMin = 0.020f;
constexpr float DetectorClosureLikeBridgeReopenFluxMax = 0.220f;
constexpr float DetectorClosureLikeBridgeReopenEvidenceMin = 0.200f;
constexpr float DetectorShortDeepValleySplitMinSec = 0.135f;
constexpr float DetectorShortDeepValleySplitMaxSec = 0.150f;
constexpr float DetectorShortDeepValleySplitMeanRMSMax = 0.020f;
constexpr float DetectorShortDeepValleySplitMinRMSMax = 0.0010f;
constexpr float DetectorShortDeepValleySplitPrevSpeechRegionMinSec = 1.500f;
constexpr float DetectorModerateGapBridgeMaxSec = 0.320f;
constexpr float DetectorModerateGapBridgeReopenFluxMax = 0.080f;
constexpr float DetectorModerateGapBridgeReopenEvidenceMax = 0.300f;
constexpr float DetectorShortIsolatedRestartSplitMinSec = 0.095f;
constexpr float DetectorShortIsolatedRestartSplitMaxSec = 0.125f;
constexpr float DetectorShortIsolatedRestartSplitMeanRMSMax = 0.050f;
constexpr float DetectorShortIsolatedRestartSplitMinRMSMax = 0.005f;
constexpr float DetectorShortIsolatedRestartSplitPrevSpeechRegionMaxSec = 0.600f;
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
constexpr float DetectorCollapsedRhetoricalSplitPrevSpeechRegionMinSec = 0.700f;
constexpr float DetectorCollapsedRhetoricalSplitReopenFluxMin = 0.120f;
constexpr float DetectorCollapsedRhetoricalSplitReopenEvidenceMin = 0.350f;
constexpr float DetectorIsolatedPulseSplitMinSec = 0.095f;
constexpr float DetectorIsolatedPulseSplitMaxSec = 0.180f;
constexpr float DetectorIsolatedPulseSplitMeanRMSMax = 0.050f;
constexpr float DetectorIsolatedPulseSplitMinRMSMax = 0.006f;
constexpr float DetectorIsolatedPulseSplitReopenFluxMin = 0.160f;
constexpr float DetectorIsolatedPulseSplitPrevSpeechRegionMaxSec = 0.650f;
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
constexpr float DetectorMicroSpeechRegionMaxDurationSec = 0.350f;
constexpr float DetectorMicroSpeechRegionPrevGapMinSec = 0.080f;
constexpr float DetectorMicroSpeechRegionNextGapMinSec = 0.180f;
constexpr float DetectorMicroSpeechRegionCombinedGapMinSec = 0.350f;
constexpr float DetectorSpeechBaselinePercentile = 0.60f;
constexpr int32 DetectorSpeechBaselineMinFrames = 12;
constexpr float DetectorSoftBridgePeriodicityMin = 0.20f;
constexpr float DetectorSoftBridgeFluxMin = 0.020f;
constexpr float DetectorSoftBridgeHighBandMin = 0.24f;
constexpr float DetectorSoftBridgeCentroidMin = 0.28f;
constexpr float DetectorSoftBridgeRMSNormMin = 0.018f;
constexpr float DetectorSoftBridgeEvidenceMin = 0.100f;
constexpr int32 DetectorListMinimumPauseFrames = 10;

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

static bool DetectorSoftCollapseBridge(
    float Evidence,
    float RMSNorm,
    float Flux,
    float Periodicity,
    float HighBandNorm,
    float SpectralCentroidNorm)
{
    const bool bVoicedBridge =
        Periodicity >= DetectorSoftBridgePeriodicityMin
        && RMSNorm >= DetectorSoftBridgeRMSNormMin;
    const bool bTurbulentBridge =
        Flux >= DetectorSoftBridgeFluxMin
        && HighBandNorm >= DetectorSoftBridgeHighBandMin
        && SpectralCentroidNorm >= DetectorSoftBridgeCentroidMin;
    const bool bBroadWeakSpeech =
        Evidence >= DetectorSoftBridgeEvidenceMin
        && RMSNorm >= DetectorSoftBridgeRMSNormMin
        && (Periodicity >= 0.12f || HighBandNorm >= 0.16f || SpectralCentroidNorm >= 0.18f);
    return bVoicedBridge || bTurbulentBridge || bBroadWeakSpeech;
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
    const FOffgridAIStreamingSpeechGapCandidate& Gap)
{
    const float GapDurationSec = Gap.GapDurationSec;
    if (GapDurationSec <= 0.055f)
    {
        return { true, FName(TEXT("micro_gap_bridge")) };
    }

    // A candidate that reopens inside the causal bridge window is not a
    // reliable speech-region boundary. Earlier feature-specific exceptions
    // split these short valleys with low precision; only persistence beyond
    // the bridge window may create a region transition.
    return { true, FName(TEXT("candidate_bridge_window_hold")) };
}
}

void FOffgridAIStreamingSpeechDetector::Reset()
{
    SpeechRegions.Reset();
    GapCandidates.Reset();
    SoftLullCandidates.Reset();
    FeatureFrames.Reset();
    LearnedSpeechRegions.Reset();
    LearnedGapCandidates.Reset();
    LearnedNextFrameIndex = 0;
    LearnedSpeechCandidateStartFrame = INDEX_NONE;
    LearnedQuietCandidateStartFrame = INDEX_NONE;
    bLearnedInSpeech = false;
    bLearnedHasObservedFirstSpeechStart = false;
    LearnedFirstSpeechAudioBufferStartSec = 0.0f;
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
    ActiveSpeechRegionPeakRMS = 0.0001f;
    ActiveSpeechRegionRMSHistory.Reset();
    ActiveSpeechRegionSeconds = 0.0f;
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
    bPendingSoftLullActive = false;
    PendingSoftLull = FOffgridAIStreamingSoftLullCandidate();
    PendingMonoSamples.Reset();
    RichAnalysisSamples.Reset();
    PreviousRichBandDistribution.Reset();
    PendingSampleBase = 0;
    ActiveSampleRate = 0;
    SpeechPeakRMS = 0.0001f;
    NoiseFloorRMS = 0.0001f;
    bListGapSensitive = false;
}

void FOffgridAIStreamingSpeechDetector::CommitPendingSoftLull()
{
    if (!bPendingSoftLullActive)
    {
        return;
    }

    if (PendingSoftLull.LullDurationSec >= DetectorAdvisorySoftLullMinSec)
    {
        if (SoftLullCandidates.Num() > 0)
        {
            FOffgridAIStreamingSoftLullCandidate& Previous = SoftLullCandidates.Last();
            const float SeparationSec = PendingSoftLull.LullStartSec - Previous.LullEndSec;
            if (Previous.SpeechRegionIndex == PendingSoftLull.SpeechRegionIndex
                && SeparationSec >= 0.0f
                && SeparationSec <= DetectorAdvisorySoftLullMergeGapSec)
            {
                Previous.LullEndSec = PendingSoftLull.LullEndSec;
                Previous.LullDurationSec = Previous.LullEndSec - Previous.LullStartSec;
                Previous.FrameCount += PendingSoftLull.FrameCount;
                Previous.RelativeRMSSum += PendingSoftLull.RelativeRMSSum;
                Previous.RelativeRMSMin = FMath::Min(Previous.RelativeRMSMin, PendingSoftLull.RelativeRMSMin);
                Previous.EvidenceSum += PendingSoftLull.EvidenceSum;
                Previous.EvidenceMin = FMath::Min(Previous.EvidenceMin, PendingSoftLull.EvidenceMin);
                Previous.ReopenEvidence = PendingSoftLull.ReopenEvidence;
                Previous.ReopenFlux = PendingSoftLull.ReopenFlux;
                Previous.bStrongOnsetReopen = PendingSoftLull.bStrongOnsetReopen;
                bPendingSoftLullActive = false;
                PendingSoftLull = FOffgridAIStreamingSoftLullCandidate();
                return;
            }
        }

        PendingSoftLull.LullIndex = SoftLullCandidates.Num();
        SoftLullCandidates.Add(PendingSoftLull);
    }
    bPendingSoftLullActive = false;
    PendingSoftLull = FOffgridAIStreamingSoftLullCandidate();
}

void FOffgridAIStreamingSpeechDetector::RefinePendingGapFromRecoveredContext(float GapEndSec)
{
    if (!bPendingGapCandidateActive || GapEndSec <= PendingGapCandidate.GapStartSec)
    {
        return;
    }

    float RefinedStartSec = PendingGapCandidate.GapStartSec;
    for (const FOffgridAIStreamingAudioFeatureFrame& Frame : FeatureFrames)
    {
        if (Frame.AudioBufferCenterSec < PendingGapCandidate.GapStartSec)
        {
            continue;
        }
        if (Frame.AudioBufferCenterSec > GapEndSec)
        {
            break;
        }
        if (Frame.SpeechEvidence <= 0.15f
            && Frame.Periodicity <= 0.25f
            && Frame.RMSNorm <= 0.02f)
        {
            RefinedStartSec = Frame.AudioBufferStartSec;
            break;
        }
    }

    if (RefinedStartSec <= PendingGapCandidate.GapStartSec + 0.001f)
    {
        return;
    }

    PendingGapCandidate.GapStartSec = RefinedStartSec;
    PendingGapCandidate.GapEndSec = GapEndSec;
    PendingGapCandidate.GapDurationSec = FMath::Max(GapEndSec - RefinedStartSec, 0.0f);
    if (SpeechRegions.IsValidIndex(PendingGapCandidate.PrevSpeechRegionIndex))
    {
        FOffgridAIStreamingSpeechRegion& Region = SpeechRegions[PendingGapCandidate.PrevSpeechRegionIndex];
        if (Region.bEnded)
        {
            Region.AudioBufferEndSec = RefinedStartSec;
        }
    }
}

void FOffgridAIStreamingSpeechDetector::SuppressRecentMicroSpeechRegionIfNeeded()
{
    if (SpeechRegions.Num() < 3)
    {
        return;
    }

    const int32 CurrentArrayIndex = SpeechRegions.Num() - 1;
    const int32 MiddleArrayIndex = CurrentArrayIndex - 1;
    const int32 PreviousArrayIndex = CurrentArrayIndex - 2;

    const FOffgridAIStreamingSpeechRegion& PreviousSpeechRegion = SpeechRegions[PreviousArrayIndex];
    const FOffgridAIStreamingSpeechRegion& MiddleSpeechRegion = SpeechRegions[MiddleArrayIndex];
    const FOffgridAIStreamingSpeechRegion& CurrentSpeechRegion = SpeechRegions[CurrentArrayIndex];

    const float MiddleDurationSec = FMath::Max(MiddleSpeechRegion.AudioBufferEndSec - MiddleSpeechRegion.AudioBufferStartSec, 0.0f);
    const float PreviousGapSec = FMath::Max(MiddleSpeechRegion.AudioBufferStartSec - PreviousSpeechRegion.AudioBufferEndSec, 0.0f);
    const float NextGapSec = FMath::Max(CurrentSpeechRegion.AudioBufferStartSec - MiddleSpeechRegion.AudioBufferEndSec, 0.0f);
    if (MiddleDurationSec > DetectorMicroSpeechRegionMaxDurationSec
        || PreviousGapSec < DetectorMicroSpeechRegionPrevGapMinSec
        || NextGapSec < DetectorMicroSpeechRegionNextGapMinSec
        || (PreviousGapSec + NextGapSec) < DetectorMicroSpeechRegionCombinedGapMinSec
        || MiddleSpeechRegion.ReopenCount > 3
        || (MiddleSpeechRegion.EndReason != FName(TEXT("strong_quiet_hangover"))
            && MiddleSpeechRegion.EndReason != FName(TEXT("weak_evidence_hangover"))))
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
            && Gap.PrevSpeechRegionIndex == MiddleSpeechRegion.SpeechRegionIndex
            && Gap.NextSpeechRegionIndex == CurrentSpeechRegion.SpeechRegionIndex)
        {
            RightGapIndex = GapIndex;
            continue;
        }
        if (LeftGapIndex == INDEX_NONE
            && !Gap.bBridged
            && Gap.PrevSpeechRegionIndex == PreviousSpeechRegion.SpeechRegionIndex
            && Gap.NextSpeechRegionIndex == MiddleSpeechRegion.SpeechRegionIndex)
        {
            LeftGapIndex = GapIndex;
        }
    }

    if (LeftGapIndex != INDEX_NONE)
    {
        FOffgridAIStreamingSpeechGapCandidate& LeftGap = GapCandidates[LeftGapIndex];
        LeftGap.GapEndSec = CurrentSpeechRegion.AudioBufferStartSec;
        LeftGap.GapDurationSec = FMath::Max(LeftGap.GapEndSec - LeftGap.GapStartSec, 0.0f);
        LeftGap.NextSpeechRegionIndex = CurrentSpeechRegion.SpeechRegionIndex;
        LeftGap.DecisionClass = FName(TEXT("candidate_micro_speech_region_suppressed"));

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

    const int32 RemovedSpeechRegionIndex = MiddleSpeechRegion.SpeechRegionIndex;
    SpeechRegions.RemoveAt(MiddleArrayIndex);
    for (int32 SpeechRegionIndex = 0; SpeechRegionIndex < SpeechRegions.Num(); ++SpeechRegionIndex)
    {
        SpeechRegions[SpeechRegionIndex].SpeechRegionIndex = SpeechRegionIndex;
    }
    for (FOffgridAIStreamingSpeechGapCandidate& Gap : GapCandidates)
    {
        if (Gap.PrevSpeechRegionIndex > RemovedSpeechRegionIndex)
        {
            Gap.PrevSpeechRegionIndex -= 1;
        }
        if (Gap.NextSpeechRegionIndex > RemovedSpeechRegionIndex)
        {
            Gap.NextSpeechRegionIndex -= 1;
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
        TArray<float> RichBandDistribution;
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
            RichBandDistribution = Distribution;
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
        RichFrame.RichBandDistribution = MoveTemp(RichBandDistribution);
        ProcessLearnedRegionFrames(false);
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
    bool bFrameStartedSpeechRegion = false;
    bool bFrameClosedSpeechRegion = false;
    bool bFrameBridgedSpeechRegion = false;
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
    const bool bSoftBridge = DetectorSoftCollapseBridge(
        Evidence,
        Frame.RMSNorm,
        Frame.Flux,
        Periodicity,
        HighBandNorm,
        SpectralCentroidNorm);
    const float RecentSpeechBaselineRMS = ActiveSpeechRegionRMSHistory.Num() >= DetectorSpeechBaselineMinFrames
        ? SpeechPercentileRMS(ActiveSpeechRegionRMSHistory, DetectorSpeechBaselinePercentile)
        : FMath::Max(ActiveSpeechRegionPeakRMS, 0.0001f);
    const float ActiveSpeechRegionRelativeRMS = RMS / FMath::Max(RecentSpeechBaselineRMS, 0.0001f);
    const bool bSoftCollapseFrame = bInSpeech
        && ActiveSpeechRegionSeconds >= 0.120f
        && ActiveSpeechRegionRelativeRMS <= DetectorSoftRelativeCollapseThreshold
        && Evidence <= DetectorSoftRelativeCollapseEvidenceMax
        && Frame.Flux <= DetectorSoftRelativeCollapseFluxMax
        && !bSoftBridge;
    const bool bSoftCollapseRelease =
        ActiveSpeechRegionRelativeRMS >= DetectorSoftRelativeCollapseReleaseThreshold
        || Evidence >= 0.18f
        || Frame.Flux >= 0.060f
        || bSoftBridge;

    const bool bRelativeCollapseFrame = bInSpeech
        && ActiveSpeechRegionSeconds >= 0.120f
        && ActiveSpeechRegionRelativeRMS <= DetectorRelativeCollapseThreshold;
    const bool bRelativeCollapseRelease = ActiveSpeechRegionRelativeRMS >= DetectorRelativeCollapseReleaseThreshold;
    const bool bHardRelativeCollapseFrame = bInSpeech
        && ActiveSpeechRegionSeconds >= 0.120f
        && ActiveSpeechRegionRelativeRMS <= DetectorHardRelativeCollapseThreshold;
    const bool bHardRelativeCollapseRelease = ActiveSpeechRegionRelativeRMS >= DetectorHardRelativeCollapseReleaseThreshold;
    const bool bStickyEndpointQuiet =
        bEndpointCandidateActive
        && ActiveSpeechRegionRelativeRMS <= DetectorStickyEndpointRelativeRMSMax
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
                const bool bCanConsiderReopen = SpeechRegions.Num() > 0 && SpeechRegions.Last().bEnded && bPendingGapCandidateActive;
                const float PendingGapAgeSec = bCanConsiderReopen
                    ? FMath::Max(SpeechCandidateStartSeconds - SpeechRegions.Last().AudioBufferEndSec, 0.0f)
                    : FLT_MAX;
                const bool bCanConsiderSoftBridge = bCanConsiderReopen
                    && PendingGapAgeSec <= DetectorSoftBridgeWindowSec
                    && SpeechCandidatePeakEvidence <= DetectorModerateGapBridgeReopenEvidenceMax
                    && Frame.Flux <= DetectorModerateGapBridgeReopenFluxMax;
                const float PendingMeanEvidence = SafeGapMean(
                    PendingGapCandidate.GapEvidenceSum,
                    PendingGapCandidate.GapFrameCount);
                const bool bCanConsiderHighFluxContinuationBridge = bCanConsiderReopen
                    && PendingGapAgeSec >= 0.200f
                    && PendingGapAgeSec <= 0.235f
                    && PendingMeanEvidence >= 0.070f
                    && Frame.Flux >= 0.250f;
                if (SpeechRegions.Num() > 0
                    && SpeechRegions.Last().bEnded
                    && (PendingGapAgeSec <= DetectorBridgeWindowSec(SpeechRegions.Last().EndReason)
                        || bCanConsiderSoftBridge
                        || bCanConsiderHighFluxContinuationBridge))
                {
                    const float PendingMeanRMSNorm = SafeGapMean(PendingGapCandidate.GapRMSNormSum, PendingGapCandidate.GapFrameCount);
                    const bool bShortIsolatedRestartOverride =
                        PendingGapCandidate.GapDurationSec >= DetectorShortIsolatedRestartSplitMinSec
                        && PendingGapCandidate.GapDurationSec <= DetectorShortIsolatedRestartSplitMaxSec
                        && PendingMeanRMSNorm <= DetectorShortIsolatedRestartSplitMeanRMSMax
                        && PendingGapCandidate.GapRMSNormMin <= DetectorShortIsolatedRestartSplitMinRMSMax
                        && PendingGapCandidate.PrevSpeechRegionDurationSec <= DetectorShortIsolatedRestartSplitPrevSpeechRegionMaxSec
                        && Frame.Flux >= DetectorShortIsolatedRestartSplitReopenFluxMin
                        && SpeechCandidatePeakEvidence >= DetectorShortIsolatedRestartSplitReopenEvidenceMin;
                    FDetectorGapDecision GapDecision = bCanConsiderHighFluxContinuationBridge
                        ? FDetectorGapDecision{ true, FName(TEXT("candidate_high_flux_continuation_bridge")) }
                        : ClassifyGapDecision(PendingGapCandidate);
                    if (GapDecision.bBridge)
                    {
                        FOffgridAIStreamingSpeechRegion& SpeechRegion = SpeechRegions.Last();
                        if (bPendingGapCandidateActive)
                        {
                            RefinePendingGapFromRecoveredContext(SpeechCandidateStartSeconds);
                            PendingGapCandidate.GapEndSec = SpeechCandidateStartSeconds;
                            PendingGapCandidate.GapDurationSec = FMath::Max(PendingGapCandidate.GapEndSec - PendingGapCandidate.GapStartSec, 0.0f);
                            PendingGapCandidate.ReopenEvidence = SpeechCandidatePeakEvidence;
                            PendingGapCandidate.ReopenFlux = Frame.Flux;
                            PendingGapCandidate.bStrongOnsetReopen = bStrongOnsetAnchor;
                            PendingGapCandidate.bBridged = true;
                            PendingGapCandidate.DecisionClass = GapDecision.DecisionClass;
                            PendingGapCandidate.NextSpeechRegionIndex = SpeechRegion.SpeechRegionIndex;
                            PendingGapCandidate.GapIndex = GapCandidates.Num();
                            GapCandidates.Add(PendingGapCandidate);
                            bPendingGapCandidateActive = false;
                        }
                        SpeechRegion.AudioBufferLastSpeechSec = FrameEndSeconds;
                        SpeechRegion.AudioBufferEndSec = FrameEndSeconds;
                        SpeechRegion.bEnded = false;
                        SpeechRegion.EndReason = FName(TEXT("reopened"));
                        SpeechRegion.ReopenCount += 1;
                        SpeechRegion.ProvisionalEndSec = -1.0f;
                        SpeechRegion.EndDecisionSec = -1.0f;
                        bFrameBridgedSpeechRegion = true;
                        OccupancyDecision = GapDecision.DecisionClass;
                    }
                    else
                    {
                        const float PrevSpeechRegionDurationSec = FMath::Max(
                            SpeechRegions.Last().AudioBufferEndSec - SpeechRegions.Last().AudioBufferStartSec,
                            0.0f);
                        const bool bReplaceLeadingBlip =
                            SpeechRegions.Num() == 1
                            && SpeechRegions.Last().AudioBufferStartSec <= 0.020f
                            && PrevSpeechRegionDurationSec <= DetectorLeadingBlipMaxDurationSec
                            && bPendingGapCandidateActive
                            && PendingGapCandidate.GapDurationSec >= DetectorLeadingBlipReplacementMinGapSec;
                        if (bReplaceLeadingBlip)
                        {
                            FOffgridAIStreamingSpeechRegion& SpeechRegion = SpeechRegions.Last();
                            SpeechRegion.AudioBufferStartSec = SpeechCandidateStartSeconds;
                            SpeechRegion.AudioBufferLastSpeechSec = FrameEndSeconds;
                            SpeechRegion.AudioBufferEndSec = FrameEndSeconds;
                            SpeechRegion.bStarted = true;
                            SpeechRegion.bEnded = false;
                            SpeechRegion.ProvisionalEndSec = -1.0f;
                            SpeechRegion.EndDecisionSec = -1.0f;
                            SpeechRegion.EndReason = NAME_None;
                            SpeechRegion.ReopenCount = 0;
                            bPendingGapCandidateActive = false;
                            bFrameStartedSpeechRegion = true;
                            OccupancyDecision = FName(TEXT("leading_blip_replaced"));
                        }
                        else
                        {
                            if (bPendingGapCandidateActive)
                            {
                                RefinePendingGapFromRecoveredContext(SpeechCandidateStartSeconds);
                                PendingGapCandidate.GapEndSec = SpeechCandidateStartSeconds;
                                PendingGapCandidate.GapDurationSec = FMath::Max(PendingGapCandidate.GapEndSec - PendingGapCandidate.GapStartSec, 0.0f);
                                PendingGapCandidate.ReopenEvidence = SpeechCandidatePeakEvidence;
                            PendingGapCandidate.ReopenFlux = Frame.Flux;
                            PendingGapCandidate.bStrongOnsetReopen = bStrongOnsetAnchor;
                            PendingGapCandidate.bBridged = false;
                            PendingGapCandidate.DecisionClass = GapDecision.DecisionClass;
                            PendingGapCandidate.NextSpeechRegionIndex = SpeechRegions.Num();
                            PendingGapCandidate.GapIndex = GapCandidates.Num();
                            GapCandidates.Add(PendingGapCandidate);
                            bPendingGapCandidateActive = false;
                        }

                            FOffgridAIStreamingSpeechRegion SpeechRegion;
                            SpeechRegion.SpeechRegionIndex = SpeechRegions.Num();
                            SpeechRegion.AudioBufferStartSec = SpeechCandidateStartSeconds;
                            SpeechRegion.AudioBufferLastSpeechSec = FrameEndSeconds;
                            SpeechRegion.AudioBufferEndSec = FrameEndSeconds;
                            SpeechRegion.bStarted = true;
                            SpeechRegions.Add(SpeechRegion);
                            SuppressRecentMicroSpeechRegionIfNeeded();
                            bFrameStartedSpeechRegion = true;
                            OccupancyDecision = GapDecision.DecisionClass;
                        }
                    }
                }
                else
                {
                    const float PrevSpeechRegionDurationSec = (SpeechRegions.Num() > 0 && SpeechRegions.Last().bEnded)
                        ? FMath::Max(SpeechRegions.Last().AudioBufferEndSec - SpeechRegions.Last().AudioBufferStartSec, 0.0f)
                        : 0.0f;
                    const bool bReplaceLeadingBlip =
                        SpeechRegions.Num() == 1
                        && SpeechRegions.Last().bEnded
                        && SpeechRegions.Last().AudioBufferStartSec <= 0.020f
                        && PrevSpeechRegionDurationSec <= DetectorLeadingBlipMaxDurationSec
                        && bPendingGapCandidateActive
                        && PendingGapCandidate.GapDurationSec >= DetectorLeadingBlipReplacementMinGapSec;
                    if (bReplaceLeadingBlip)
                    {
                        FOffgridAIStreamingSpeechRegion& SpeechRegion = SpeechRegions.Last();
                        SpeechRegion.AudioBufferStartSec = SpeechCandidateStartSeconds;
                        SpeechRegion.AudioBufferLastSpeechSec = FrameEndSeconds;
                        SpeechRegion.AudioBufferEndSec = FrameEndSeconds;
                        SpeechRegion.bStarted = true;
                        SpeechRegion.bEnded = false;
                        SpeechRegion.ProvisionalEndSec = -1.0f;
                        SpeechRegion.EndDecisionSec = -1.0f;
                        SpeechRegion.EndReason = NAME_None;
                        SpeechRegion.ReopenCount = 0;
                        bPendingGapCandidateActive = false;
                        bFrameStartedSpeechRegion = true;
                        OccupancyDecision = FName(TEXT("leading_blip_replaced"));
                    }
                    else
                    {
                        if (bCanConsiderReopen && bPendingGapCandidateActive)
                        {
                            RefinePendingGapFromRecoveredContext(SpeechCandidateStartSeconds);
                            PendingGapCandidate.GapEndSec = SpeechCandidateStartSeconds;
                            PendingGapCandidate.GapDurationSec = FMath::Max(PendingGapCandidate.GapEndSec - PendingGapCandidate.GapStartSec, 0.0f);
                            PendingGapCandidate.ReopenEvidence = SpeechCandidatePeakEvidence;
                            PendingGapCandidate.ReopenFlux = Frame.Flux;
                            PendingGapCandidate.bStrongOnsetReopen = bStrongOnsetAnchor;
                            PendingGapCandidate.bBridged = false;
                            PendingGapCandidate.DecisionClass = FName(TEXT("bridge_window_expired_split"));
                            PendingGapCandidate.NextSpeechRegionIndex = SpeechRegions.Num();
                            PendingGapCandidate.GapIndex = GapCandidates.Num();
                            GapCandidates.Add(PendingGapCandidate);
                            bPendingGapCandidateActive = false;
                        }
                        FOffgridAIStreamingSpeechRegion SpeechRegion;
                        SpeechRegion.SpeechRegionIndex = SpeechRegions.Num();
                        SpeechRegion.AudioBufferStartSec = SpeechCandidateStartSeconds;
                        SpeechRegion.AudioBufferLastSpeechSec = FrameEndSeconds;
                        SpeechRegion.AudioBufferEndSec = FrameEndSeconds;
                        SpeechRegion.bStarted = true;
                        SpeechRegions.Add(SpeechRegion);
                        SuppressRecentMicroSpeechRegionIfNeeded();
                        bFrameStartedSpeechRegion = true;
                        OccupancyDecision = bCanConsiderReopen ? FName(TEXT("bridge_window_expired_split")) : FName(TEXT("new_speech_region_open"));
                    }
                }
                bInSpeech = true;
                bSpeechCandidateActive = false;
                SpeechCandidatePeakEvidence = 0.0f;
                ActiveSpeechRegionPeakRMS = RMS;
                ActiveSpeechRegionRMSHistory.Reset();
                ActiveSpeechRegionRMSHistory.Add(RMS);
                ActiveSpeechRegionSeconds = 0.0f;
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
        FOffgridAIStreamingSpeechRegion& SpeechRegion = SpeechRegions.Last();
        SpeechRegion.AudioBufferEndSec = FrameEndSeconds;
        ActiveSpeechRegionPeakRMS = FMath::Max(ActiveSpeechRegionPeakRMS, RMS);
        if (bKeepOpen && Evidence >= 0.16f && RMS >= CloseThreshold)
        {
            ActiveSpeechRegionRMSHistory.Add(RMS);
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
            SpeechRegion.AudioBufferLastSpeechSec = FrameEndSeconds;
            ActiveSpeechRegionSeconds += FrameEndSeconds - FrameStartSeconds;
            if (bEndpointCandidateActive)
            {
                EndpointCandidateMinEvidence = FMath::Min(EndpointCandidateMinEvidence, Evidence);
                SpeechRegion.ReopenCount += 1;
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
                    SpeechRegion.ProvisionalEndSec = EndpointCandidateStartSeconds;
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
                SpeechRegion.ProvisionalEndSec = -1.0f;
                OccupancyDecision = bSoftBridge
                    ? FName(TEXT("audible_bridge_keep_open"))
                    : FName(TEXT("keep_open"));
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
                SpeechRegion.ProvisionalEndSec = SilenceStartSeconds;
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
                SpeechRegion.AudioBufferEndSec = EndpointCandidateStartSeconds;
                SpeechRegion.EndDecisionSec = FrameEndSeconds;
                if (bSustainedHardRelativeCollapse)
                {
                    SpeechRegion.EndReason = FName(TEXT("hard_relative_collapse_hangover"));
                }
                else if (bSustainedRelativeCollapse)
                {
                    SpeechRegion.EndReason = FName(TEXT("relative_collapse_hangover"));
                }
                else if (bSustainedSoftCollapse)
                {
                    SpeechRegion.EndReason = FName(TEXT("soft_collapse_hangover"));
                }
                else
                {
                    SpeechRegion.EndReason = bStrongQuiet ? FName(TEXT("strong_quiet_hangover")) : FName(TEXT("weak_evidence_hangover"));
                }
                PendingGapCandidate = FOffgridAIStreamingSpeechGapCandidate();
                PendingGapCandidate.PrevSpeechRegionIndex = SpeechRegion.SpeechRegionIndex;
                PendingGapCandidate.GapStartSec = EndpointCandidateStartSeconds;
                PendingGapCandidate.PrevSpeechRegionDurationSec = FMath::Max(SpeechRegion.AudioBufferEndSec - SpeechRegion.AudioBufferStartSec, 0.0f);
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
                PendingGapCandidate.CloseReason = SpeechRegion.EndReason;
                bPendingGapCandidateActive = true;
                SpeechRegion.bEnded = true;
                bFrameClosedSpeechRegion = true;
                OccupancyDecision = SpeechRegion.EndReason;
                bInSpeech = false;
                SilenceAccumSeconds = 0.0f;
                ActiveSpeechRegionRMSHistory.Reset();
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
    Frame.ActiveSpeechRegionStartSec = SpeechRegions.Num() > 0 ? SpeechRegions.Last().AudioBufferStartSec : -1.0f;
    Frame.ActiveSpeechRegionEndSec = SpeechRegions.Num() > 0 ? SpeechRegions.Last().AudioBufferEndSec : -1.0f;
    Frame.bInSpeechBeforeFrame = bWasInSpeechAtFrameStart;
    Frame.bInSpeechAfterFrame = bInSpeech;
    Frame.bOpenCandidate = bOpen;
    Frame.bKeepOpen = bKeepOpen;
    Frame.bStrongOnsetAnchor = bStrongOnsetAnchor;
    Frame.bStrongQuiet = bStrongQuiet;
    Frame.bLowEvidence = bLowEvidence;
    Frame.bEndpointCandidateActive = bEndpointCandidateActive;
    Frame.bFrameStartedSpeechRegion = bFrameStartedSpeechRegion;
    Frame.bFrameClosedSpeechRegion = bFrameClosedSpeechRegion;
    Frame.bFrameBridgedSpeechRegion = bFrameBridgedSpeechRegion;
    Frame.OccupancyDecision = OccupancyDecision;

    // Endpoint candidates include short low-energy valleys that occupancy may
    // correctly bridge. Preserve them separately for pause/resume advisories.
    if (Frame.bEndpointCandidateActive)
    {
        if (!bPendingSoftLullActive)
        {
            bPendingSoftLullActive = true;
            PendingSoftLull = FOffgridAIStreamingSoftLullCandidate();
            PendingSoftLull.SpeechRegionIndex = SpeechRegions.Num() > 0
                ? SpeechRegions.Last().SpeechRegionIndex
                : INDEX_NONE;
            PendingSoftLull.LullStartSec = Frame.EndpointCandidateStartSec >= 0.0f
                ? Frame.EndpointCandidateStartSec
                : FrameStartSeconds;
        }
        PendingSoftLull.LullEndSec = FrameEndSeconds;
        PendingSoftLull.LullDurationSec = PendingSoftLull.LullEndSec - PendingSoftLull.LullStartSec;
        PendingSoftLull.FrameCount += 1;
        PendingSoftLull.RelativeRMSSum += ActiveSpeechRegionRelativeRMS;
        PendingSoftLull.RelativeRMSMin = FMath::Min(PendingSoftLull.RelativeRMSMin, ActiveSpeechRegionRelativeRMS);
        PendingSoftLull.EvidenceSum += Evidence;
        PendingSoftLull.EvidenceMin = FMath::Min(PendingSoftLull.EvidenceMin, Evidence);
    }
    else if (bPendingSoftLullActive)
    {
        PendingSoftLull.ReopenEvidence = Evidence;
        PendingSoftLull.ReopenFlux = Frame.Flux;
        PendingSoftLull.bStrongOnsetReopen = bStrongOnsetAnchor;
        CommitPendingSoftLull();
    }

    Frame.bListGapSensitive = bListGapSensitive;
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

float FOffgridAIStreamingSpeechDetector::ComputeLearnedSpeechProbability(int32 FrameIndex) const
{
    constexpr int32 BaseFeatureCount = 10;
    float Base[BaseFeatureCount] = {};
    auto FillBase = [](const FOffgridAIStreamingAudioFeatureFrame& Frame, float* Out)
    {
        Out[0] = std::log(FMath::Max(Frame.RMS, 1.0e-6f));
        Out[1] = Frame.RMSNorm;
        Out[2] = Frame.DeltaRMS;
        Out[3] = Frame.Flux;
        Out[4] = Frame.ZCR;
        Out[5] = Frame.LowBandNorm;
        Out[6] = Frame.MidBandNorm;
        Out[7] = Frame.HighBandNorm;
        Out[8] = Frame.SpectralCentroidNorm;
        Out[9] = Frame.Periodicity;
    };
    FillBase(FeatureFrames[FrameIndex], Base);

    float Features[StreamingRegionFeatureCount] = {};
    for (int32 FeatureIndex = 0; FeatureIndex < BaseFeatureCount; ++FeatureIndex)
    {
        Features[FeatureIndex] = Base[FeatureIndex];
    }

    const int32 WindowBegins[3] = {
        FMath::Max(FrameIndex - 10, 0),
        FrameIndex,
        FMath::Max(FrameIndex - 10, 0),
    };
    const int32 WindowEnds[3] = {
        FrameIndex,
        FMath::Min(FrameIndex + 10, FeatureFrames.Num() - 1),
        FMath::Min(FrameIndex + 10, FeatureFrames.Num() - 1),
    };
    for (int32 WindowIndex = 0; WindowIndex < 3; ++WindowIndex)
    {
        const int32 Begin = WindowBegins[WindowIndex];
        const int32 End = WindowEnds[WindowIndex];
        const float Denominator = static_cast<float>(FMath::Max(End - Begin + 1, 1));
        for (int32 SampleIndex = Begin; SampleIndex <= End; ++SampleIndex)
        {
            float Sample[BaseFeatureCount] = {};
            FillBase(FeatureFrames[SampleIndex], Sample);
            for (int32 FeatureIndex = 0; FeatureIndex < BaseFeatureCount; ++FeatureIndex)
            {
                Features[(WindowIndex + 1) * BaseFeatureCount + FeatureIndex] +=
                    Sample[FeatureIndex] / Denominator;
            }
        }
    }

    float Logit = StreamingRegionBias;
    for (int32 FeatureIndex = 0; FeatureIndex < StreamingRegionFeatureCount; ++FeatureIndex)
    {
        Logit += Features[FeatureIndex] * StreamingRegionWeights[FeatureIndex];
    }
    return 1.0f / (1.0f + std::exp(-FMath::Clamp(Logit, -30.0f, 30.0f)));
}

void FOffgridAIStreamingSpeechDetector::DecodeLearnedSpeechFrame(int32 FrameIndex)
{
    FOffgridAIStreamingAudioFeatureFrame& Frame = FeatureFrames[FrameIndex];
    Frame.LearnedSpeechProbability = ComputeLearnedSpeechProbability(FrameIndex);
    Frame.bLearnedSpeech = Frame.LearnedSpeechProbability >= StreamingRegionSpeechThreshold;

    if (!bLearnedInSpeech)
    {
        if (!Frame.bLearnedSpeech)
        {
            LearnedSpeechCandidateStartFrame = INDEX_NONE;
            return;
        }
        if (LearnedSpeechCandidateStartFrame == INDEX_NONE)
        {
            LearnedSpeechCandidateStartFrame = FrameIndex;
        }
        if (FrameIndex - LearnedSpeechCandidateStartFrame + 1 < StreamingRegionMinimumSpeechFrames)
        {
            return;
        }

        FOffgridAIStreamingSpeechRegion Region;
        Region.SpeechRegionIndex = LearnedSpeechRegions.Num();
        Region.AudioBufferStartSec = FeatureFrames[LearnedSpeechCandidateStartFrame].AudioBufferStartSec;
        Region.AudioBufferLastSpeechSec = Frame.AudioBufferEndSec;
        Region.AudioBufferEndSec = Frame.AudioBufferEndSec;
        Region.bStarted = true;
        LearnedSpeechRegions.Add(Region);

        if (LearnedSpeechRegions.Num() > 1)
        {
            const FOffgridAIStreamingSpeechRegion& Previous = LearnedSpeechRegions[LearnedSpeechRegions.Num() - 2];
            FOffgridAIStreamingSpeechGapCandidate Gap;
            Gap.GapIndex = LearnedGapCandidates.Num();
            Gap.PrevSpeechRegionIndex = Previous.SpeechRegionIndex;
            Gap.NextSpeechRegionIndex = Region.SpeechRegionIndex;
            Gap.GapStartSec = Previous.AudioBufferEndSec;
            Gap.GapEndSec = Region.AudioBufferStartSec;
            Gap.GapDurationSec = FMath::Max(Gap.GapEndSec - Gap.GapStartSec, 0.0f);
            Gap.bBridged = false;
            Gap.CloseReason = Previous.EndReason;
            Gap.DecisionClass = FName(TEXT("learned_duration_split"));
            LearnedGapCandidates.Add(Gap);
        }

        if (!bLearnedHasObservedFirstSpeechStart)
        {
            bLearnedHasObservedFirstSpeechStart = true;
            LearnedFirstSpeechAudioBufferStartSec = Region.AudioBufferStartSec;
        }
        bLearnedInSpeech = true;
        LearnedSpeechCandidateStartFrame = INDEX_NONE;
        LearnedQuietCandidateStartFrame = INDEX_NONE;
        return;
    }

    FOffgridAIStreamingSpeechRegion& Region = LearnedSpeechRegions.Last();
    if (Frame.bLearnedSpeech)
    {
        LearnedQuietCandidateStartFrame = INDEX_NONE;
        Region.AudioBufferLastSpeechSec = Frame.AudioBufferEndSec;
        Region.AudioBufferEndSec = Frame.AudioBufferEndSec;
        return;
    }

    if (LearnedQuietCandidateStartFrame == INDEX_NONE)
    {
        LearnedQuietCandidateStartFrame = FrameIndex;
    }
    const bool bListSensitiveQuietRun =
        FeatureFrames[LearnedQuietCandidateStartFrame].bListGapSensitive;
    const int32 MinimumPauseFrames = bListSensitiveQuietRun
        ? DetectorListMinimumPauseFrames
        : StreamingRegionMinimumPauseFrames;
    if (FrameIndex - LearnedQuietCandidateStartFrame + 1 < MinimumPauseFrames)
    {
        return;
    }

    const float CloseSec = FeatureFrames[LearnedQuietCandidateStartFrame].AudioBufferStartSec;
    Region.AudioBufferEndSec = CloseSec;
    Region.ProvisionalEndSec = CloseSec;
    Region.EndDecisionSec = Frame.AudioBufferEndSec;
    Region.bEnded = true;
    Region.EndReason = FName(TEXT("learned_duration_close"));
    bLearnedInSpeech = false;
    LearnedQuietCandidateStartFrame = INDEX_NONE;
    LearnedSpeechCandidateStartFrame = INDEX_NONE;
}

void FOffgridAIStreamingSpeechDetector::ProcessLearnedRegionFrames(bool bFlush)
{
    const int32 FinalizableCount = bFlush
        ? FeatureFrames.Num()
        : FMath::Max(FeatureFrames.Num() - 10, 0);
    while (LearnedNextFrameIndex < FinalizableCount)
    {
        DecodeLearnedSpeechFrame(LearnedNextFrameIndex);
        ++LearnedNextFrameIndex;
    }
}

void FOffgridAIStreamingSpeechDetector::Finalize(float FinalObservedAudioBufferEndSec)
{
    ProcessLearnedRegionFrames(true);
    if (LearnedSpeechRegions.Num() > 0 && bLearnedInSpeech)
    {
        FOffgridAIStreamingSpeechRegion& Region = LearnedSpeechRegions.Last();
        const float CloseSec = LearnedQuietCandidateStartFrame != INDEX_NONE
            ? FeatureFrames[LearnedQuietCandidateStartFrame].AudioBufferStartSec
            : Region.AudioBufferLastSpeechSec;
        Region.AudioBufferEndSec = CloseSec;
        Region.ProvisionalEndSec = CloseSec;
        Region.EndDecisionSec = FinalObservedAudioBufferEndSec >= 0.0f
            ? FinalObservedAudioBufferEndSec
            : ObservedAudioBufferEndSec;
        Region.bEnded = true;
        Region.EndReason = FName(TEXT("learned_finalize"));
        bLearnedInSpeech = false;
    }
    if (bPendingSoftLullActive)
    {
        CommitPendingSoftLull();
    }
    if (SpeechRegions.Num() > 0 && bInSpeech)
    {
        FOffgridAIStreamingSpeechRegion& SpeechRegion = SpeechRegions.Last();
        SpeechRegion.bEnded = true;

        // Finalization is not fresh acoustic speech. If an endpoint candidate is
        // already active, close at the first quiet frame. Otherwise close at the
        // last frame that actually met the keep-open speech criteria. This keeps
        // trailing buffering/final-drain mechanics out of detector speech-region
        // scoring.
        if (bEndpointCandidateActive && EndpointCandidateStartSeconds > SpeechRegion.AudioBufferStartSec)
        {
            SpeechRegion.AudioBufferEndSec = EndpointCandidateStartSeconds;
            SpeechRegion.ProvisionalEndSec = EndpointCandidateStartSeconds;
            SpeechRegion.EndDecisionSec = FinalObservedAudioBufferEndSec >= 0.0f ? FinalObservedAudioBufferEndSec : ObservedAudioBufferEndSec;
            SpeechRegion.EndReason = FName(TEXT("finalize_at_provisional_end"));
            PendingGapCandidate = FOffgridAIStreamingSpeechGapCandidate();
            PendingGapCandidate.PrevSpeechRegionIndex = SpeechRegion.SpeechRegionIndex;
            PendingGapCandidate.GapStartSec = EndpointCandidateStartSeconds;
            PendingGapCandidate.PrevSpeechRegionDurationSec = FMath::Max(SpeechRegion.AudioBufferEndSec - SpeechRegion.AudioBufferStartSec, 0.0f);
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
            PendingGapCandidate.CloseReason = SpeechRegion.EndReason;
            bPendingGapCandidateActive = true;
        }
        else if (SpeechRegion.AudioBufferLastSpeechSec > SpeechRegion.AudioBufferStartSec)
        {
            SpeechRegion.AudioBufferEndSec = SpeechRegion.AudioBufferLastSpeechSec;
            SpeechRegion.EndDecisionSec = FinalObservedAudioBufferEndSec >= 0.0f ? FinalObservedAudioBufferEndSec : ObservedAudioBufferEndSec;
            SpeechRegion.EndReason = FName(TEXT("finalize_at_last_speech"));
        }
        else
        {
            SpeechRegion.AudioBufferEndSec = FinalObservedAudioBufferEndSec >= 0.0f ? FinalObservedAudioBufferEndSec : ObservedAudioBufferEndSec;
            SpeechRegion.EndDecisionSec = SpeechRegion.AudioBufferEndSec;
            SpeechRegion.EndReason = FName(TEXT("finalize_no_candidate"));
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
