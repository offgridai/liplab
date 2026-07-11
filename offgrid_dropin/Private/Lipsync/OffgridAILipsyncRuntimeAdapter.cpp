#include "Lipsync/OffgridAILipsyncRuntimeAdapter.h"

namespace
{
static const FName TextPriorMonotonicCommitReason(TEXT("text_prior_monotonic_commit"));

// Advisory landmark extraction/logging now lives in LineCoach's consolidated
// debug landmark system. The runtime adapter must not use landmarks/prosody
// to move the text playhead; start/resume acoustic anchors are the only audio
// timing authority.
static constexpr float MaxLiveCommitLeadSec = 0.320f;
static constexpr float MaxLiveCommitBehindSec = 0.180f;

static constexpr float InterWordSpacerSeconds = 0.020f;
static constexpr float ActiveDurationScale = 0.90f;

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

struct FEffectiveSpeechRegion
{
    float StartSec = 0.0f;
    float EndSec = 0.0f;
};

static bool IsHardPausePunctuation(TCHAR C)
{
    return C == TEXT('.')
        || C == TEXT('!')
        || C == TEXT('?')
        || C == TEXT(':')
        || C == TEXT(';')
        || C == TEXT('-')
        || C == TEXT('—')
        || C == TEXT('–');
}

static bool IsHardLikeBoundaryClass(TCHAR C, EOffgridAIBoundaryPauseClass PauseClass)
{
    return PauseClass == EOffgridAIBoundaryPauseClass::HardBreakPause || IsHardPausePunctuation(C);
}

static FString BoundaryPauseClassToString(EOffgridAIBoundaryPauseClass PauseClass)
{
    switch (PauseClass)
    {
    case EOffgridAIBoundaryPauseClass::None:
        return TEXT("None");
    case EOffgridAIBoundaryPauseClass::SoftListPause:
        return TEXT("SoftListPause");
    case EOffgridAIBoundaryPauseClass::HardBreakPause:
        return TEXT("HardBreakPause");
    default:
        return TEXT("Unknown");
    }
}

static bool IsSoftListBoundaryClass(TCHAR C, EOffgridAIBoundaryPauseClass PauseClass)
{
    return C == TEXT(',') && PauseClass == EOffgridAIBoundaryPauseClass::SoftListPause;
}

static float HoldSecondsForBoundaryClass(TCHAR C, EOffgridAIBoundaryPauseClass PauseClass)
{
    // Punctuation patience is derived from the text boundary class. A comma
    // in a list context is low-patience and may be spoken through; a comma that
    // separates clauses is classed as HardBreakPause and gets period-like
    // patience.  Hard punctuation marks are always high-patience.
    if (IsSoftListBoundaryClass(C, PauseClass))
    {
        return 0.420f;
    }
    if (IsHardLikeBoundaryClass(C, PauseClass))
    {
        // Text-prior pacing can arrive at a sentence fence well before the TTS
        // audio when the preceding clause contains several cadence pauses. Give
        // hard punctuation enough time to observe the real close/resume rather
        // than declaring continuous speech just before the sentence gap arrives.
        return 1.150f;
    }
    return 0.0f;
}

static float MinCloseConfidenceForBoundaryClass(TCHAR C, EOffgridAIBoundaryPauseClass PauseClass)
{
    if (IsSoftListBoundaryClass(C, PauseClass))
    {
        return 0.42f;
    }
    return IsHardLikeBoundaryClass(C, PauseClass) ? 0.58f : 0.50f;
}

static float MinGapDurationForBoundaryClass(TCHAR C, EOffgridAIBoundaryPauseClass PauseClass)
{
    if (IsSoftListBoundaryClass(C, PauseClass))
    {
        return 0.025f;
    }
    return IsHardLikeBoundaryClass(C, PauseClass) ? 0.060f : 0.035f;
}

static float MinResumeConfidenceForBoundaryClass(TCHAR C, EOffgridAIBoundaryPauseClass PauseClass)
{
    return IsSoftListBoundaryClass(C, PauseClass) ? 0.34f : 0.40f;
}

static float MinSoftLullConfidenceForBoundaryClass(TCHAR C, EOffgridAIBoundaryPauseClass PauseClass)
{
    if (IsHardLikeBoundaryClass(C, PauseClass))
    {
        return 0.62f;
    }
    return IsSoftListBoundaryClass(C, PauseClass) ? 0.40f : 0.56f;
}

static int32 ResumeStableRunFramesForBoundaryClass(TCHAR C, EOffgridAIBoundaryPauseClass PauseClass)
{
    return IsSoftListBoundaryClass(C, PauseClass) ? 3 : 4;
}

static float LiveResumeSearchLookaheadForBoundaryClass(TCHAR C, EOffgridAIBoundaryPauseClass PauseClass)
{
    if (IsHardLikeBoundaryClass(C, PauseClass))
    {
        return 0.180f;
    }
    return IsSoftListBoundaryClass(C, PauseClass) ? 0.120f : 0.060f;
}

static float ResumeBloomRawRMSForBoundaryClass(
    TCHAR C,
    EOffgridAIBoundaryPauseClass PauseClass,
    float QuietRawRMSAtDecay)
{
    const float Multiplier = IsSoftListBoundaryClass(C, PauseClass) ? 2.4f : 3.0f;
    return FMath::Max(0.004f, QuietRawRMSAtDecay * Multiplier + 0.001f);
}

static float MinDecayToResumeGapForBoundaryClass(TCHAR C, EOffgridAIBoundaryPauseClass PauseClass)
{
    return IsHardLikeBoundaryClass(C, PauseClass) ? 0.120f : 0.020f;
}

static float MinVisibleGapForBoundaryClass(TCHAR C, EOffgridAIBoundaryPauseClass PauseClass)
{
    if (IsSoftListBoundaryClass(C, PauseClass))
    {
        return 0.140f;
    }
    return IsHardLikeBoundaryClass(C, PauseClass) ? 0.135f : 0.030f;
}

static float MaxArtificialResumeDelayForBoundaryClass(TCHAR C, EOffgridAIBoundaryPauseClass PauseClass)
{
    return IsHardLikeBoundaryClass(C, PauseClass) ? 0.160f : 0.260f;
}

static float FallbackNoDecayPauseSecondsForBoundaryClass(TCHAR C, EOffgridAIBoundaryPauseClass PauseClass)
{
    // Spoken-through list commas often still carry a small cadence delay even
    // when no stable out-of-speech / resume pair is observed. Preserve that
    // rhythm instead of collapsing to zero pause.
    return IsSoftListBoundaryClass(C, PauseClass) ? 0.180f : 0.0f;
}

static int32 FindFeatureFrameIndexAtPlayback(
    const TArray<FOffgridAIStreamingAudioFeatureFrame>* AudioFeatureFrames,
    float PlaybackSec)
{
    if (!AudioFeatureFrames || AudioFeatureFrames->Num() <= 0)
    {
        return INDEX_NONE;
    }

    for (int32 Index = AudioFeatureFrames->Num() - 1; Index >= 0; --Index)
    {
        const FOffgridAIStreamingAudioFeatureFrame& Frame = (*AudioFeatureFrames)[Index];
        if (PlaybackSec >= Frame.AudioBufferStartSec)
        {
            return Index;
        }
    }

    return 0;
}

static bool HasStableRawSpeechRunFromFrame(
    const TArray<FOffgridAIStreamingAudioFeatureFrame>* AudioFeatureFrames,
    int32 StartFrameIndex,
    int32 RequiredFrameCount,
    float MinRawRMS)
{
    if (!AudioFeatureFrames || StartFrameIndex < 0 || RequiredFrameCount <= 0)
    {
        return false;
    }

    if (StartFrameIndex + RequiredFrameCount > AudioFeatureFrames->Num())
    {
        return false;
    }

    for (int32 Offset = 0; Offset < RequiredFrameCount; ++Offset)
    {
        const FOffgridAIStreamingAudioFeatureFrame& Frame = (*AudioFeatureFrames)[StartFrameIndex + Offset];
        if (Frame.RMS < MinRawRMS)
        {
            return false;
        }

        const bool bSpeechLike = Frame.bInSpeechAfterFrame
            || Frame.bOpenCandidate
            || Frame.bStrongOnsetAnchor
            || Frame.SpeechEvidence >= FMath::Max(Frame.OpenThreshold, 0.18f);
        if (!bSpeechLike)
        {
            return false;
        }
    }

    return true;
}



static bool IsRawQuietPunctuationDecayFrame(const FOffgridAIStreamingAudioFeatureFrame& Frame)
{
    // Boundary decay must be grounded in the actual PCM floor. The adaptive
    // detector can briefly report low evidence / strong quiet inside a weak
    // consonant or phrase tail; accepting that as punctuation decay lets the
    // next word resume immediately inside the same utterance.
    constexpr float MaxQuietRawRMS = 0.0035f;
    if (Frame.RMS > MaxQuietRawRMS)
    {
        return false;
    }

    return Frame.bStrongQuiet
        || Frame.bLowEvidence
        || Frame.bFrameClosedSpeechRegion
        || (!Frame.bInSpeechAfterFrame && Frame.SpeechEvidence <= 0.30f);
}

static bool HasStableRawQuietRunEndingAtFrame(
    const TArray<FOffgridAIStreamingAudioFeatureFrame>* AudioFeatureFrames,
    int32 EndFrameIndex,
    int32 RequiredFrameCount,
    float HoldStartPlaybackSec)
{
    if (!AudioFeatureFrames || EndFrameIndex < 0 || RequiredFrameCount <= 0)
    {
        return false;
    }

    const int32 FirstFrameIndex = EndFrameIndex - RequiredFrameCount + 1;
    if (FirstFrameIndex < 0)
    {
        return false;
    }

    for (int32 Index = FirstFrameIndex; Index <= EndFrameIndex; ++Index)
    {
        const FOffgridAIStreamingAudioFeatureFrame& Frame = (*AudioFeatureFrames)[Index];
        if (Frame.AudioBufferCenterSec + 0.001f < HoldStartPlaybackSec)
        {
            return false;
        }
        if (!IsRawQuietPunctuationDecayFrame(Frame))
        {
            return false;
        }
    }

    return true;
}


static bool IsConfirmedOutOfSpeechForHardPause(const FOffgridAIStreamingAudioFeatureFrame* Frame);

static bool HasStableHardPauseQuietRunEndingAtFrame(
    const TArray<FOffgridAIStreamingAudioFeatureFrame>* AudioFeatureFrames,
    int32 EndFrameIndex,
    int32 RequiredFrameCount,
    float HoldStartPlaybackSec)
{
    if (!AudioFeatureFrames || EndFrameIndex < 0 || RequiredFrameCount <= 0)
    {
        return false;
    }

    const int32 FirstFrameIndex = EndFrameIndex - RequiredFrameCount + 1;
    if (FirstFrameIndex < 0)
    {
        return false;
    }

    for (int32 Index = FirstFrameIndex; Index <= EndFrameIndex; ++Index)
    {
        const FOffgridAIStreamingAudioFeatureFrame& Frame = (*AudioFeatureFrames)[Index];
        if (Frame.AudioBufferCenterSec + 0.001f < HoldStartPlaybackSec)
        {
            return false;
        }
        if (!IsConfirmedOutOfSpeechForHardPause(&Frame))
        {
            return false;
        }
    }

    return true;
}


static bool IsPunctuationPauseSpeechFrame(const FOffgridAIStreamingAudioFeatureFrame* Frame)
{
    if (!Frame)
    {
        return false;
    }

    return Frame->bInSpeechAfterFrame
        && !Frame->bStrongQuiet
        && (Frame->SpeechEvidence >= 0.38f || Frame->RMSNorm >= 0.20f);
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

static int32 FindContainingRegionIndexAtPlayback(const TArray<FEffectiveSpeechRegion>& Regions, float PlaybackSec)
{
    for (int32 RegionIndex = 0; RegionIndex < Regions.Num(); ++RegionIndex)
    {
        const FEffectiveSpeechRegion& Region = Regions[RegionIndex];
        if (PlaybackSec >= Region.StartSec && PlaybackSec <= Region.EndSec)
        {
            return RegionIndex;
        }
    }
    return INDEX_NONE;
}

static bool IsConfirmedOutOfSpeechForHardPause(const FOffgridAIStreamingAudioFeatureFrame* Frame)
{
    if (!Frame)
    {
        return false;
    }

    if (Frame->bFrameClosedSpeechRegion || Frame->bStrongQuiet)
    {
        return true;
    }

    return !Frame->bInSpeechAfterFrame
        && (Frame->bLowEvidence
            || Frame->SpeechEvidence <= 0.24f
            || (Frame->RMSNorm <= 0.13f && Frame->SpeechEvidence <= 0.30f));
}

static bool IsFreshResumeOnsetFrame(const FOffgridAIStreamingAudioFeatureFrame* Frame)
{
    if (!Frame)
    {
        return false;
    }

    if (!IsPunctuationPauseSpeechFrame(Frame))
    {
        return false;
    }

    return Frame->bFrameStartedSpeechRegion
        || Frame->bStrongOnsetAnchor
        || (!Frame->bInSpeechBeforeFrame && Frame->bInSpeechAfterFrame)
        || (Frame->bOpenCandidate && Frame->SpeechEvidence >= 0.38f)
        || (Frame->Flux >= 0.12f && Frame->SpeechEvidence >= 0.38f);
}

static bool IsStableResumeAnchorFrame(
    const FOffgridAIStreamingAudioFeatureFrame* Frame,
    TCHAR BoundaryMark,
    EOffgridAIBoundaryPauseClass PauseClass,
    float QuietRMSNormAtDecay,
    float QuietEvidenceAtDecay,
    float QuietRawRMSAtDecay)
{
    if (!Frame || !IsPunctuationPauseSpeechFrame(Frame))
    {
        return false;
    }

    // Anchor acceptance must be grounded in raw audio level, not only detector
    // normalized evidence. RMSNorm can be high during near-silence because it is
    // adaptive; raw RMS prevents punctuation/initial anchors from accepting
    // the first few noise frames or a tiny transient blip.
    // Use a principled bloom threshold: resumed speech must rise clearly above
    // the raw quiet floor captured at the punctuation decay.  This avoids both
    // hard-coded fake offsets and accepting tiny post-pause blips as anchors.
    const float ResumeBloomRawRMS =
        ResumeBloomRawRMSForBoundaryClass(BoundaryMark, PauseClass, QuietRawRMSAtDecay);
    if (Frame->RMS < ResumeBloomRawRMS)
    {
        return false;
    }

    const bool bDetectorOpen = Frame->bInSpeechAfterFrame || Frame->bOpenCandidate || Frame->bStrongOnsetAnchor;
    const bool bEvidenceStable = Frame->SpeechEvidence >= FMath::Max(Frame->OpenThreshold, 0.20f)
        || (Frame->bStrongOnsetAnchor && Frame->SpeechEvidence >= 0.16f);
    const bool bEnergyStable = Frame->RMSNorm >= FMath::Max(QuietRMSNormAtDecay + 0.020f, 0.080f)
        || Frame->RMS >= FMath::Max(0.006f, ResumeBloomRawRMS)
        || (Frame->Flux >= 0.10f && Frame->SpeechEvidence >= Frame->OpenThreshold && Frame->RMS >= ResumeBloomRawRMS)
        || (Frame->bStrongOnsetAnchor && Frame->RMS >= ResumeBloomRawRMS);

    return bDetectorOpen && bEvidenceStable && bEnergyStable;
}


static bool FindBufferedBoundaryResumeAnchorSec(
    const TArray<FOffgridAIStreamingAudioFeatureFrame>* AudioFeatureFrames,
    TCHAR BoundaryMark,
    EOffgridAIBoundaryPauseClass PauseClass,
    float HoldStartPlaybackSec,
    float MaxSearchPlaybackSec,
    float MinDecayToResumeGapSec,
    float& OutQuietStartSec,
    float& OutResumeOnsetSec,
    float& OutResumeAnchorSec,
    float& OutQuietRMSNorm,
    float& OutQuietEvidence,
    float& OutQuietRawRMS)
{
    if (!AudioFeatureFrames || AudioFeatureFrames->Num() <= 0)
    {
        return false;
    }

    int32 QuietFrameIndex = INDEX_NONE;
    for (int32 FrameIndex = 0; FrameIndex < AudioFeatureFrames->Num(); ++FrameIndex)
    {
        const FOffgridAIStreamingAudioFeatureFrame& Frame = (*AudioFeatureFrames)[FrameIndex];
        if (Frame.AudioBufferCenterSec + 0.001f < HoldStartPlaybackSec)
        {
            continue;
        }
        if (Frame.AudioBufferCenterSec > MaxSearchPlaybackSec + 0.001f)
        {
            break;
        }

        const bool bQuietRun = IsHardLikeBoundaryClass(BoundaryMark, PauseClass)
            ? HasStableHardPauseQuietRunEndingAtFrame(AudioFeatureFrames, FrameIndex, 3, HoldStartPlaybackSec)
            : HasStableRawQuietRunEndingAtFrame(AudioFeatureFrames, FrameIndex, 3, HoldStartPlaybackSec);
        if (bQuietRun)
        {
            QuietFrameIndex = FMath::Max(FrameIndex - 2, 0);
            const FOffgridAIStreamingAudioFeatureFrame& QuietFrame = (*AudioFeatureFrames)[QuietFrameIndex];
            OutQuietStartSec = QuietFrame.AudioBufferCenterSec;
            OutQuietRMSNorm = QuietFrame.RMSNorm;
            OutQuietEvidence = QuietFrame.SpeechEvidence;
            OutQuietRawRMS = QuietFrame.RMS;
            break;
        }
    }

    if (QuietFrameIndex == INDEX_NONE)
    {
        return false;
    }

    for (int32 FrameIndex = QuietFrameIndex + 1; FrameIndex < AudioFeatureFrames->Num(); ++FrameIndex)
    {
        const FOffgridAIStreamingAudioFeatureFrame& Frame = (*AudioFeatureFrames)[FrameIndex];
        if (Frame.AudioBufferCenterSec > MaxSearchPlaybackSec + 0.001f)
        {
            break;
        }
        if (Frame.AudioBufferCenterSec + 0.001f < OutQuietStartSec + MinDecayToResumeGapSec)
        {
            continue;
        }

        const bool bStableRawSpeechRun = HasStableRawSpeechRunFromFrame(
            AudioFeatureFrames,
            FrameIndex,
            ResumeStableRunFramesForBoundaryClass(BoundaryMark, PauseClass),
            ResumeBloomRawRMSForBoundaryClass(BoundaryMark, PauseClass, OutQuietRawRMS));
        if (IsStableResumeAnchorFrame(&Frame, BoundaryMark, PauseClass, OutQuietRMSNorm, OutQuietEvidence, OutQuietRawRMS)
            && bStableRawSpeechRun)
        {
            OutResumeOnsetSec = Frame.AudioBufferCenterSec;
            OutResumeAnchorSec = Frame.AudioBufferCenterSec;
            return true;
        }
    }

    return false;
}

struct FCausalBoundaryFenceEstimate
{
    bool bObservedClose = false;
    bool bObservedResume = false;
    bool bContinuousSpeech = false;
    bool bUsedSoftLull = false;
    float CloseSec = -1.0f;
    float ResumeOnsetSec = -1.0f;
    float ResumeAnchorSec = -1.0f;
    float QuietRMSNorm = 1.0f;
    float QuietEvidence = 1.0f;
    float QuietRawRMS = 1.0f;
    FName Outcome = NAME_None;
};

static float GapCloseConfidence(const FOffgridAIStreamingSpeechGapCandidate& Gap)
{
    const float MeanEvidence = Gap.GapFrameCount > 0
        ? Gap.GapEvidenceSum / Gap.GapFrameCount
        : Gap.QuietEvidence;
    const float MeanRMSNorm = Gap.GapFrameCount > 0
        ? Gap.GapRMSNormSum / Gap.GapFrameCount
        : Gap.QuietRMSNorm;
    const float QuietSupport = FMath::Clamp((0.26f - Gap.QuietEvidence) / 0.26f, 0.0f, 1.0f);
    const float MeanQuietSupport = FMath::Clamp((0.24f - MeanEvidence) / 0.24f, 0.0f, 1.0f);
    const float QuietFloor = FMath::Clamp((0.12f - Gap.QuietRMSNorm) / 0.12f, 0.0f, 1.0f);
    const float MeanFloor = FMath::Clamp((0.14f - MeanRMSNorm) / 0.14f, 0.0f, 1.0f);
    const float DurationSupport = FMath::Clamp((Gap.GapDurationSec - 0.030f) / 0.130f, 0.0f, 1.0f);
    const float LowEvidenceRatio = Gap.GapFrameCount > 0
        ? static_cast<float>(Gap.LowEvidenceFrameCount) / Gap.GapFrameCount
        : 0.0f;
    const float StrongQuietRatio = Gap.GapFrameCount > 0
        ? static_cast<float>(Gap.StrongQuietFrameCount) / Gap.GapFrameCount
        : 0.0f;
    const float ClassBonus =
        Gap.DecisionClass == FName(TEXT("bridge_window_expired_split")) ? 0.10f :
        Gap.DecisionClass == FName(TEXT("short_isolated_restart_split")) ? 0.08f :
        Gap.DecisionClass == FName(TEXT("collapsed_rhetorical_split")) ? 0.08f :
        Gap.DecisionClass == FName(TEXT("isolated_pulse_split")) ? 0.06f :
        Gap.bBridged ? 0.03f : 0.0f;

    return FMath::Clamp(
        QuietSupport * 0.24f +
        MeanQuietSupport * 0.18f +
        QuietFloor * 0.18f +
        MeanFloor * 0.10f +
        DurationSupport * 0.12f +
        LowEvidenceRatio * 0.10f +
        StrongQuietRatio * 0.05f +
        (Gap.bStrongQuietClose ? 0.08f : 0.0f) +
        ClassBonus,
        0.0f,
        1.0f);
}

static float GapResumeConfidence(const FOffgridAIStreamingSpeechGapCandidate& Gap)
{
    const float DurationSupport = FMath::Clamp((Gap.GapDurationSec - 0.030f) / 0.130f, 0.0f, 1.0f);
    const float ReopenSupport = FMath::Clamp((Gap.ReopenEvidence - 0.20f) / 0.30f, 0.0f, 1.0f);
    const float ReopenFluxSupport = FMath::Clamp((Gap.ReopenFlux - 0.04f) / 0.20f, 0.0f, 1.0f);
    const float QuietSupport = FMath::Clamp((0.26f - Gap.QuietEvidence) / 0.26f, 0.0f, 1.0f);
    const float ClassBonus =
        Gap.DecisionClass == FName(TEXT("bridge_window_expired_split")) ? 0.10f :
        Gap.DecisionClass == FName(TEXT("short_isolated_restart_split")) ? 0.08f :
        Gap.DecisionClass == FName(TEXT("collapsed_rhetorical_split")) ? 0.08f :
        Gap.DecisionClass == FName(TEXT("isolated_pulse_split")) ? 0.06f :
        Gap.bBridged ? 0.03f : 0.0f;

    return FMath::Clamp(
        ReopenSupport * 0.34f +
        ReopenFluxSupport * 0.20f +
        DurationSupport * 0.10f +
        QuietSupport * 0.10f +
        (Gap.bStrongOnsetReopen ? 0.16f : 0.0f) +
        ClassBonus,
        0.0f,
        1.0f);
}

static float SoftLullConfidence(const FOffgridAIStreamingSoftLullCandidate& Lull)
{
    if (Lull.FrameCount <= 0)
    {
        return 0.0f;
    }

    const float MeanRelativeRMS = Lull.RelativeRMSSum / Lull.FrameCount;
    const float MeanEvidence = Lull.EvidenceSum / Lull.FrameCount;
    const float Depth = FMath::Clamp((0.24f - MeanRelativeRMS) / 0.20f, 0.0f, 1.0f);
    const float Floor = FMath::Clamp((0.18f - Lull.RelativeRMSMin) / 0.16f, 0.0f, 1.0f);
    const float Quiet = FMath::Clamp((0.16f - MeanEvidence) / 0.14f, 0.0f, 1.0f);
    const float Duration = FMath::Clamp((Lull.LullDurationSec - 0.025f) / 0.095f, 0.0f, 1.0f);
    const float Reopen = FMath::Clamp((Lull.ReopenEvidence - 0.12f) / 0.30f, 0.0f, 1.0f);
    // A breath or low-level nonverbal can keep detector evidence elevated while
    // the waveform itself has plainly collapsed. Reward a sustained near-floor
    // energy valley so an early, deep punctuation pause is not rejected in
    // favor of a later and shallower word-internal lull.
    const float DeepCollapse =
        FMath::Clamp((0.08f - MeanRelativeRMS) / 0.06f, 0.0f, 1.0f) *
        FMath::Clamp((0.04f - Lull.RelativeRMSMin) / 0.03f, 0.0f, 1.0f);

    return FMath::Clamp(
        Depth * 0.20f + Floor * 0.18f + Quiet * 0.10f + Duration * 0.28f +
        Reopen * 0.16f + DeepCollapse * 0.08f +
        (Lull.bStrongOnsetReopen ? 0.12f : 0.0f),
        0.0f,
        1.0f);
}

static bool FindStableResumeAnchorAfterTime(
    const TArray<FOffgridAIStreamingAudioFeatureFrame>* AudioFeatureFrames,
    TCHAR BoundaryMark,
    EOffgridAIBoundaryPauseClass PauseClass,
    float QuietStartSec,
    float QuietRMSNorm,
    float QuietEvidence,
    float QuietRawRMS,
    float MinResumeSec,
    float MaxSearchSec,
    float& OutResumeOnsetSec,
    float& OutResumeAnchorSec)
{
    if (!AudioFeatureFrames || AudioFeatureFrames->Num() <= 0)
    {
        return false;
    }

    for (int32 FrameIndex = 0; FrameIndex < AudioFeatureFrames->Num(); ++FrameIndex)
    {
        const FOffgridAIStreamingAudioFeatureFrame& Frame = (*AudioFeatureFrames)[FrameIndex];
        if (Frame.AudioBufferCenterSec + 0.001f < MinResumeSec)
        {
            continue;
        }
        if (Frame.AudioBufferCenterSec > MaxSearchSec + 0.001f)
        {
            break;
        }

        if (!IsStableResumeAnchorFrame(&Frame, BoundaryMark, PauseClass, QuietRMSNorm, QuietEvidence, QuietRawRMS))
        {
            continue;
        }

        const bool bStableRawSpeechRun = HasStableRawSpeechRunFromFrame(
            AudioFeatureFrames,
            FrameIndex,
            ResumeStableRunFramesForBoundaryClass(BoundaryMark, PauseClass),
            ResumeBloomRawRMSForBoundaryClass(BoundaryMark, PauseClass, QuietRawRMS));
        if (!bStableRawSpeechRun)
        {
            continue;
        }

        OutResumeOnsetSec = Frame.AudioBufferCenterSec;
        OutResumeAnchorSec = Frame.AudioBufferCenterSec;
        return true;
    }

    return false;
}

static FCausalBoundaryFenceEstimate EvaluateCausalBoundaryFence(
    const FOffgridAILipsyncRuntimeUpdateInput& Input,
    const FOffgridAIBoundaryPlaybackState& State,
    float PlaybackSec)
{
    FCausalBoundaryFenceEstimate Estimate;
    const bool bHardBoundary = IsHardLikeBoundaryClass(State.ActiveBoundaryMark, State.ActivePauseClass);
    const float HoldWindowSec = HoldSecondsForBoundaryClass(State.ActiveBoundaryMark, State.ActivePauseClass);
    const float MaxObservedSearchSec = Input.bPlaybackFinalized
        ? TNumericLimits<float>::Max()
        : FMath::Min(
            Input.ObservedAudioBufferEndSec,
            PlaybackSec + LiveResumeSearchLookaheadForBoundaryClass(State.ActiveBoundaryMark, State.ActivePauseClass));

    const float MinResumeGapSec = MinDecayToResumeGapForBoundaryClass(State.ActiveBoundaryMark, State.ActivePauseClass);

    if (Input.GapCandidates)
    {
        for (const FOffgridAIStreamingSpeechGapCandidate& Gap : *Input.GapCandidates)
        {
            if (Gap.GapDurationSec <= 0.0f)
            {
                continue;
            }
            // BoundarySearchStartPlaybackSec already includes the permitted
            // 80ms look-behind into the preceding word's decay. A candidate
            // that began materially before it belongs to an older boundary;
            // allowing it merely because its tail overlaps this search lets one
            // long gap get reused for several commas in a list.
            if (Gap.GapStartSec + 0.020f < State.BoundarySearchStartPlaybackSec)
            {
                continue;
            }
            if (Gap.GapStartSec > MaxObservedSearchSec + 0.001f)
            {
                break;
            }

            const float CloseConfidence = GapCloseConfidence(Gap);
            const float ResumeConfidence = GapResumeConfidence(Gap);
            const bool bCloseAccepted =
                CloseConfidence >= MinCloseConfidenceForBoundaryClass(State.ActiveBoundaryMark, State.ActivePauseClass)
                && Gap.GapDurationSec >= MinGapDurationForBoundaryClass(State.ActiveBoundaryMark, State.ActivePauseClass)
                && (!bHardBoundary || !Gap.bBridged);
            if (!bCloseAccepted)
            {
                continue;
            }

            Estimate.bObservedClose = true;
            Estimate.CloseSec = Gap.GapStartSec + FMath::Min(0.060f, FMath::Max(0.020f, Gap.GapDurationSec * 0.20f));
            Estimate.QuietRMSNorm = Gap.QuietRMSNorm;
            Estimate.QuietEvidence = Gap.QuietEvidence;
            Estimate.QuietRawRMS = 0.001f;
            Estimate.Outcome = FName(TEXT("pause_close_candidate"));

            if (ResumeConfidence >= MinResumeConfidenceForBoundaryClass(State.ActiveBoundaryMark, State.ActivePauseClass)
                && Gap.GapEndSec + 0.001f >= Estimate.CloseSec + MinResumeGapSec)
            {
                float ResumeOnsetSec = -1.0f;
                float ResumeAnchorSec = -1.0f;
                const float QuietRawRMS = FMath::Max(0.001f, Gap.GapRMSNormMin * 0.05f);
                if (FindStableResumeAnchorAfterTime(
                        Input.AudioFeatureFrames,
                        State.ActiveBoundaryMark,
                        State.ActivePauseClass,
                        Estimate.CloseSec,
                        Gap.QuietRMSNorm,
                        Gap.QuietEvidence,
                        QuietRawRMS,
                        Gap.GapEndSec,
                        MaxObservedSearchSec,
                        ResumeOnsetSec,
                        ResumeAnchorSec))
                {
                    Estimate.bObservedResume = true;
                    Estimate.ResumeOnsetSec = ResumeOnsetSec;
                    Estimate.ResumeAnchorSec = ResumeAnchorSec;
                    Estimate.QuietRawRMS = QuietRawRMS;
                    Estimate.Outcome = FName(TEXT("confirmed_resume_anchor_gap"));
                }
            }

            if (Estimate.bObservedResume)
            {
                return Estimate;
            }

            // A speech-gap candidate can be very strong evidence of closure
            // while its coarse reopen fields remain too weak to authorize a
            // resume. Keep the close, but let the finer soft-lull path inspect
            // the same neighborhood instead of shadowing it until finalization.
            break;
        }
    }

    if (Input.SoftLullCandidates)
    {
        for (const FOffgridAIStreamingSoftLullCandidate& Lull : *Input.SoftLullCandidates)
        {
            // Hard punctuation should not attach to the short cadence valleys
            // that occur naturally inside the preceding sentence. Gold speech
            // regions coalesce sub-120ms gaps; requiring most of that support
            // here still admits a causal 90ms detector candidate while filtering
            // the 50-80ms bridged dips used for soft comma rhythm.
            const float MinLullDurationSec = bHardBoundary ? 0.090f : 0.030f;
            if (Lull.LullDurationSec < MinLullDurationSec || Lull.FrameCount <= 0)
            {
                continue;
            }
            if (Lull.LullStartSec + 0.020f < State.BoundarySearchStartPlaybackSec)
            {
                continue;
            }
            if (Lull.LullStartSec > MaxObservedSearchSec + 0.001f)
            {
                break;
            }

            const float Confidence = SoftLullConfidence(Lull);
            if (Confidence < MinSoftLullConfidenceForBoundaryClass(State.ActiveBoundaryMark, State.ActivePauseClass))
            {
                continue;
            }

            Estimate.bObservedClose = true;
            Estimate.bUsedSoftLull = true;
            Estimate.CloseSec = Lull.LullStartSec + FMath::Min(0.050f, Lull.LullDurationSec);
            Estimate.QuietRMSNorm = Lull.RelativeRMSMin;
            Estimate.QuietEvidence = Lull.FrameCount > 0 ? (Lull.EvidenceSum / Lull.FrameCount) : 0.0f;
            Estimate.QuietRawRMS = 0.001f;
            Estimate.Outcome = FName(TEXT("pause_close_soft_lull"));

            float ResumeOnsetSec = -1.0f;
            float ResumeAnchorSec = -1.0f;
            if (FindStableResumeAnchorAfterTime(
                    Input.AudioFeatureFrames,
                    State.ActiveBoundaryMark,
                    State.ActivePauseClass,
                    Estimate.CloseSec,
                    Estimate.QuietRMSNorm,
                    Estimate.QuietEvidence,
                    Estimate.QuietRawRMS,
                    Lull.LullEndSec,
                    MaxObservedSearchSec,
                    ResumeOnsetSec,
                    ResumeAnchorSec))
            {
                Estimate.bObservedResume = true;
                Estimate.ResumeOnsetSec = ResumeOnsetSec;
                Estimate.ResumeAnchorSec = ResumeAnchorSec;
                Estimate.Outcome = FName(TEXT("confirmed_resume_anchor_soft_lull"));
            }

            return Estimate;
        }
    }

    // Once a deadline-time tail check has confirmed that audio is becoming
    // quiet, keep the boundary fence open and seek the corresponding resume.
    // Do not let the ordinary patience timeout relabel that known close as
    // continuous speech on the next tick.
    if (State.ConfirmedQuietStartPlaybackSec >= 0.0f)
    {
        Estimate.bObservedClose = true;
        Estimate.CloseSec = State.ConfirmedQuietStartPlaybackSec;
        Estimate.QuietRMSNorm = State.QuietRMSNormAtDecay;
        Estimate.QuietEvidence = State.QuietEvidenceAtDecay;
        Estimate.QuietRawRMS = State.QuietRawRMSAtDecay;
        Estimate.Outcome = FName(TEXT("pause_close_pending_resume"));

        float ResumeOnsetSec = -1.0f;
        float ResumeAnchorSec = -1.0f;
        if (FindStableResumeAnchorAfterTime(
                Input.AudioFeatureFrames,
                State.ActiveBoundaryMark,
                State.ActivePauseClass,
                Estimate.CloseSec,
                Estimate.QuietRMSNorm,
                Estimate.QuietEvidence,
                Estimate.QuietRawRMS,
                Estimate.CloseSec + MinResumeGapSec,
                MaxObservedSearchSec,
                ResumeOnsetSec,
                ResumeAnchorSec))
        {
            Estimate.bObservedResume = true;
            Estimate.ResumeOnsetSec = ResumeOnsetSec;
            Estimate.ResumeAnchorSec = ResumeAnchorSec;
            Estimate.Outcome = FName(TEXT("confirmed_resume_anchor_pending_close"));
        }
        return Estimate;
    }

    const bool bCoveredPatienceWindow =
        Input.bPlaybackFinalized
        || (HoldWindowSec > 0.0f && Input.ObservedAudioBufferEndSec + 0.001f >= State.HoldStartPlaybackSec + HoldWindowSec);
    if (bCoveredPatienceWindow && PlaybackSec >= State.HoldDeadlinePlaybackSec)
    {
        if (bHardBoundary && Input.AudioFeatureFrames && Input.AudioFeatureFrames->Num() > 0)
        {
            const int32 TailFrameIndex = FindFeatureFrameIndexAtPlayback(
                Input.AudioFeatureFrames,
                Input.ObservedAudioBufferEndSec);
            if (HasStableHardPauseQuietRunEndingAtFrame(
                    Input.AudioFeatureFrames,
                    TailFrameIndex,
                    3,
                    State.BoundarySearchStartPlaybackSec))
            {
                const int32 QuietFrameIndex = FMath::Max(TailFrameIndex - 2, 0);
                const FOffgridAIStreamingAudioFeatureFrame& QuietFrame = (*Input.AudioFeatureFrames)[QuietFrameIndex];
                Estimate.bObservedClose = true;
                Estimate.CloseSec = QuietFrame.AudioBufferCenterSec;
                Estimate.QuietRMSNorm = QuietFrame.RMSNorm;
                Estimate.QuietEvidence = QuietFrame.SpeechEvidence;
                Estimate.QuietRawRMS = QuietFrame.RMS;
                Estimate.Outcome = FName(TEXT("pause_close_buffer_tail"));
                return Estimate;
            }
        }

        Estimate.bContinuousSpeech = true;
        Estimate.Outcome = FName(TEXT("no_decay_continuous"));
    }

    return Estimate;
}

static bool IsInitialStableSpeechAnchorFrame(const FOffgridAIStreamingAudioFeatureFrame& Frame)
{
    // Initial start anchors must be based on real PCM energy. Detector-open and
    // normalized evidence alone can mark the first buffered frame as speech even
    // when raw audio is near-silent, especially after region backdating.
    constexpr float MinInitialAnchorRawRMS = 0.006f;
    if (Frame.RMS < MinInitialAnchorRawRMS)
    {
        return false;
    }

    const bool bDetectorOpen = Frame.bInSpeechAfterFrame || Frame.bOpenCandidate || Frame.bStrongOnsetAnchor;
    const bool bEvidenceStable = Frame.SpeechEvidence >= FMath::Max(Frame.OpenThreshold, 0.22f)
        || (Frame.bStrongOnsetAnchor && Frame.SpeechEvidence >= 0.18f);
    return bDetectorOpen && bEvidenceStable;
}

static bool FindInitialSpeechEnergyAnchorSec(
    const TArray<FOffgridAIStreamingAudioFeatureFrame>* Frames,
    const TArray<FEffectiveSpeechRegion>& EffectiveRegions,
    float& OutAnchorSec)
{
    OutAnchorSec = -1.0f;
    if (!Frames || Frames->Num() <= 0 || EffectiveRegions.Num() <= 0)
    {
        return false;
    }

    const float FirstRegionStartSec = EffectiveRegions[0].StartSec;
    int32 CandidateIndex = INDEX_NONE;
    int32 StableRunFrames = 0;
    for (int32 FrameIndex = 0; FrameIndex < Frames->Num(); ++FrameIndex)
    {
        const FOffgridAIStreamingAudioFeatureFrame& Frame = (*Frames)[FrameIndex];
        if (Frame.AudioBufferEndSec + 0.001f < FirstRegionStartSec)
        {
            continue;
        }

        if (IsInitialStableSpeechAnchorFrame(Frame))
        {
            if (CandidateIndex == INDEX_NONE)
            {
                CandidateIndex = FrameIndex;
            }
            ++StableRunFrames;

            // Require sustained raw acoustic evidence for initial start.  Region-open
            // and single-frame onset flags are too permissive at t=0 and were
            // still producing bogus 0.005s anchors on low-energy lead-in noise.
            if (StableRunFrames >= 4)
            {
                // Anchor to the confirmed incoming energy bloom frame, not the
                // first candidate frame.  If speech starts immediately at t=0,
                // anchoring the candidate gives the repeated bogus 0.005s start.
                OutAnchorSec = Frame.AudioBufferCenterSec;
                return true;
            }
        }
        else
        {
            CandidateIndex = INDEX_NONE;
            StableRunFrames = 0;
        }
    }
    return false;
}

static void AdvancePlaybackHoldState(
    const FOffgridAILipsyncRuntimeUpdateInput& Input,
    const TArray<FEffectiveSpeechRegion>& EffectiveRegions,
    FOffgridAIBoundaryPlaybackState& InOutState)
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

    auto ReanchorPausedClockToAcousticAnchor = [&](float AcousticAnchorSec)
    {
        // Boundary timing has three independent facts:
        // 1) the pre-boundary phrase has already committed and owns its visible tail,
        // 2) audio supplies the first stable resumed speech,
        // 3) the post-boundary phrase starts no earlier than a small visible gap
        //    after the prior tail, with bounded artificial delay.
        // This avoids both old failure modes: post-boundary overlap, and hunting
        // for a much later energy peak that makes the resumed word arrive late.
        const bool bHardBoundary = IsHardLikeBoundaryClass(InOutState.ActiveBoundaryMark, InOutState.ActivePauseClass);
        const float AcousticBaseSec = (bHardBoundary && InOutState.ObservedResumeOnsetPlaybackSec >= 0.0f)
            ? InOutState.ObservedResumeOnsetPlaybackSec
            : AcousticAnchorSec;
        const float MinVisibleGapSec = MinVisibleGapForBoundaryClass(InOutState.ActiveBoundaryMark, InOutState.ActivePauseClass);
        const float MaxArtificialDelaySec = MaxArtificialResumeDelayForBoundaryClass(InOutState.ActiveBoundaryMark, InOutState.ActivePauseClass);
        float DesiredCenterSec = FMath::Max(AcousticBaseSec, 0.0f);
        DesiredCenterSec = FMath::Min(DesiredCenterSec, AcousticBaseSec + MaxArtificialDelaySec);
        if (InOutState.HoldPreviousCommittedRenderEndSec >= 0.0f)
        {
            const float EarliestVisibleCenterSec =
                InOutState.HoldPreviousCommittedRenderEndSec
                + MinVisibleGapSec
                + InOutState.HoldResumeTargetLeadSec;
            DesiredCenterSec = FMath::Max(DesiredCenterSec, EarliestVisibleCenterSec);
        }
        DesiredCenterSec = FMath::Max(DesiredCenterSec, 0.0f);

        const float DesiredPausedSec =
            DesiredCenterSec + InOutState.HoldResumeTargetLeadSec
            - InOutState.PlaybackOriginSec
            - InOutState.HoldResumeTargetActiveSec;
        InOutState.TotalPausedSec = FMath::Max(DesiredPausedSec, 0.0f);

        // A confirmed punctuation resume starts a new local text-prior segment.
        // Do not carry active-playhead debt from the pre-boundary segment across
        // the pause: long comma/list runs can otherwise leave the virtual text
        // clock ahead of the first post-boundary event, causing the resumed
        // suffix to be dumped or visually starved.  The committed prefix is
        // immutable; the suffix now advances from the first post-boundary
        // event's active time as its fresh origin.
        InOutState.ActivePlayheadSec = InOutState.HoldResumeTargetActiveSec;

        InOutState.bResumeAnchorActive = true;
        InOutState.bResumeAnchorFromInitialSpeech = false;
        InOutState.ResumeAnchorActiveSec = InOutState.HoldResumeTargetActiveSec;
        InOutState.ResumeAnchorLeadSec = InOutState.HoldResumeTargetLeadSec;
        InOutState.ResumeAnchorFinalCenterSec = DesiredCenterSec;
        InOutState.ObservedResumeEnergyAnchorSec = DesiredCenterSec;
        InOutState.LastResolvedBoundary.DecaySec = InOutState.ConfirmedQuietStartPlaybackSec;
        InOutState.LastResolvedBoundary.ResumeOnsetSec = InOutState.ObservedResumeOnsetPlaybackSec;
        InOutState.LastResolvedBoundary.ResumeEnergyAnchorSec = DesiredCenterSec;
    };

    bool bReleasedContinuousNoDecayThisTick = false;

    if (InOutState.bHoldActive)
    {
        FCausalBoundaryFenceEstimate Fence = EvaluateCausalBoundaryFence(Input, InOutState, PlaybackSec);
        if (!Fence.bObservedResume && Input.bPlaybackFinalized)
        {
            float BufferedQuietSec = -1.0f;
            float BufferedResumeOnsetSec = -1.0f;
            float BufferedResumeAnchorSec = -1.0f;
            float BufferedQuietRMSNorm = 1.0f;
            float BufferedQuietEvidence = 1.0f;
            float BufferedQuietRawRMS = 1.0f;
            const bool bFoundBufferedResume = FindBufferedBoundaryResumeAnchorSec(
                Input.AudioFeatureFrames,
                InOutState.ActiveBoundaryMark,
                InOutState.ActivePauseClass,
                InOutState.BoundarySearchStartPlaybackSec,
                TNumericLimits<float>::Max(),
                MinDecayToResumeGapForBoundaryClass(InOutState.ActiveBoundaryMark, InOutState.ActivePauseClass),
                BufferedQuietSec,
                BufferedResumeOnsetSec,
                BufferedResumeAnchorSec,
                BufferedQuietRMSNorm,
                BufferedQuietEvidence,
                BufferedQuietRawRMS);
            if (bFoundBufferedResume)
            {
                Fence.bObservedClose = true;
                Fence.bObservedResume = true;
                Fence.bContinuousSpeech = false;
                Fence.CloseSec = BufferedQuietSec;
                Fence.ResumeOnsetSec = BufferedResumeOnsetSec;
                Fence.ResumeAnchorSec = BufferedResumeAnchorSec;
                Fence.QuietRMSNorm = BufferedQuietRMSNorm;
                Fence.QuietEvidence = BufferedQuietEvidence;
                Fence.QuietRawRMS = BufferedQuietRawRMS;
                Fence.Outcome = FName(TEXT("confirmed_resume_anchor_final_search"));
            }
            else if (!Fence.bContinuousSpeech)
            {
                Fence.bContinuousSpeech = true;
                Fence.Outcome = Fence.bObservedClose
                    ? FName(TEXT("finalized_abandoned"))
                    : FName(TEXT("no_decay_continuous"));
            }
        }
        InOutState.LastFenceEstimatorOutcome = Fence.Outcome;

        if (Fence.bObservedClose)
        {
            InOutState.bObservedPauseLull = true;
            InOutState.bSawConfirmedOutOfSpeechAfterBoundary = true;
            if (InOutState.ConfirmedQuietStartPlaybackSec < 0.0f)
            {
                InOutState.ConfirmedQuietStartPlaybackSec = Fence.CloseSec;
                InOutState.QuietRMSNormAtDecay = Fence.QuietRMSNorm;
                InOutState.QuietEvidenceAtDecay = Fence.QuietEvidence;
                InOutState.QuietRawRMSAtDecay = Fence.QuietRawRMS;
            }
        }

        if (Fence.bObservedResume)
        {
            InOutState.ObservedResumeOnsetPlaybackSec = Fence.ResumeOnsetSec;
            InOutState.ObservedResumeEnergyAnchorSec = Fence.ResumeAnchorSec;
            InOutState.QuietRMSNormAtDecay = Fence.QuietRMSNorm;
            InOutState.QuietEvidenceAtDecay = Fence.QuietEvidence;
            InOutState.QuietRawRMSAtDecay = Fence.QuietRawRMS;
            ReanchorPausedClockToAcousticAnchor(Fence.ResumeAnchorSec);
            InOutState.LastResolvedBoundary.Outcome = Fence.Outcome;
            InOutState.LastResolvedBoundary.WordIndex = InOutState.BoundaryWordIndex;
            InOutState.LastResolvedBoundary.Mark = InOutState.ActiveBoundaryMark;
            InOutState.bHoldActive = false;
            InOutState.bObservedPauseLull = false;
            InOutState.bSawConfirmedOutOfSpeechAfterBoundary = false;
            InOutState.ConfirmedQuietStartPlaybackSec = -1.0f;
            InOutState.QuietRMSNormAtDecay = 1.0f;
            InOutState.QuietEvidenceAtDecay = 1.0f;
            InOutState.QuietRawRMSAtDecay = 1.0f;
            InOutState.ActivePauseClass = EOffgridAIBoundaryPauseClass::None;
            InOutState.ActiveBoundaryMark = TCHAR(0);
        }
        else if (Fence.bContinuousSpeech)
        {
            InOutState.TotalPausedSec += FallbackNoDecayPauseSecondsForBoundaryClass(
                InOutState.ActiveBoundaryMark,
                InOutState.ActivePauseClass);
            InOutState.ActivePlayheadSec += FMath::Max(PlaybackSec - InOutState.HoldStartPlaybackSec, 0.0f);
            InOutState.LastResolvedBoundary.Outcome = Fence.Outcome;
            InOutState.LastResolvedBoundary.WordIndex = InOutState.BoundaryWordIndex;
            InOutState.LastResolvedBoundary.Mark = InOutState.ActiveBoundaryMark;
            InOutState.bResumeAnchorActive = false;
            InOutState.bResumeAnchorFromInitialSpeech = false;
            InOutState.ResumeAnchorEventIndex = INDEX_NONE;
            InOutState.ResumeAnchorFinalCenterSec = 0.0f;
            InOutState.ObservedResumeOnsetPlaybackSec = -1.0f;
            InOutState.ObservedResumeEnergyAnchorSec = -1.0f;
            InOutState.LastResolvedBoundary.DecaySec = -1.0f;
            InOutState.LastResolvedBoundary.ResumeOnsetSec = -1.0f;
            InOutState.LastResolvedBoundary.ResumeEnergyAnchorSec = -1.0f;
            InOutState.bHoldActive = false;
            InOutState.bObservedPauseLull = false;
            InOutState.bSawConfirmedOutOfSpeechAfterBoundary = false;
            InOutState.ConfirmedQuietStartPlaybackSec = -1.0f;
            InOutState.QuietRMSNormAtDecay = 1.0f;
            InOutState.QuietEvidenceAtDecay = 1.0f;
            InOutState.QuietRawRMSAtDecay = 1.0f;
            InOutState.ActivePauseClass = EOffgridAIBoundaryPauseClass::None;
            InOutState.ActiveBoundaryMark = TCHAR(0);
            bReleasedContinuousNoDecayThisTick = true;
        }
    }

    if (!InOutState.bHoldActive && !bReleasedContinuousNoDecayThisTick)
    {
        InOutState.ActivePlayheadSec += DeltaSec;
    }

    InOutState.LastPlaybackSec = PlaybackSec;
}

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

static float ComputeFirstSpeechStart(const TArray<FEffectiveSpeechRegion>& Regions)
{
    return Regions.Num() > 0 ? Regions[0].StartSec : -1.0f;
}

static float ComputeLastSpeechEnd(const TArray<FEffectiveSpeechRegion>& Regions, float ObservedEndSec)
{
    return Regions.Num() > 0 ? Regions.Last().EndSec : ObservedEndSec;
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

static int32 ComputeNextPlanEventIndex(const FOffgridAICommittedVisemeTrack& Track)
{
    int32 MaxSeenEventIndex = INDEX_NONE;
    for (const FOffgridAICommittedVisemeEvent& E : Track.Events)
    {
        MaxSeenEventIndex = FMath::Max(MaxSeenEventIndex, E.EventIndex);
    }
    return MaxSeenEventIndex + 1;
}

static float ComputeLastCommittedCenterSec(const FOffgridAICommittedVisemeTrack& Track)
{
    float LastCenterSec = -1.0f;
    for (const FOffgridAICommittedVisemeEvent& E : Track.Events)
    {
        LastCenterSec = FMath::Max(LastCenterSec, E.FinalRenderCenterSeconds);
    }
    return LastCenterSec;
}

static void FillEventFromText(
    const FOffgridAITextVisemePlan& Plan,
    int32 EventIndex,
    const FOffgridAITextVisemeEvent& T,
    float OrderNorm,
    float Center,
    float Span,
    const FOffgridAILipsyncRuntimeUpdateInput& Input,
    FOffgridAICommittedVisemeEvent& Out)
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
    Out.CommitReason = TextPriorMonotonicCommitReason;

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
    ResolvedSpeechRegions.Reset();
    CommittedTrack = FOffgridAICommittedVisemeTrack();
    RuntimeCommitDiagnosticRows.Reset();
    RuntimeSpeechRegionDiagnosticRows.Reset();
    RuntimeCommitDiagnosticUpdateOrdinal = 0;
    StreamTailDiagnosticRow = FOffgridAIStreamTailDiagnosticRow();

    PCMChunkCount = 0;
    PCMBytesReceived = 0;
    PCMSamplesReceived = 0;
    LastPCMChunkSampleRate = 0;
    LastPCMChunkChannels = 0;
    LastPCMChunkStartSample = -1;
    LastPCMChunkEndSample = -1;
    PunctuationHoldState = FOffgridAIBoundaryPlaybackState();
}

void FOffgridAILipsyncRuntimeSession::BeginLine(const FOffgridAILipsyncRuntimeBeginInput& Input)
{
    Reset();
    NPCID = Input.NPCID;
    LineID = Input.LineID;
    DialogueText = Input.DialogueText;
    PrerollSec = FMath::Max(Input.PrerollSec, 0.0f);
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
    Input.GapCandidates = &Detector.GetGapCandidates();
    Input.SoftLullCandidates = &Detector.GetSoftLullCandidates();
    Input.AudioFeatureFrames = &Detector.GetFeatureFrames();
    Input.CurrentPlaybackSec = PlaybackSec;
    Input.PrerollSec = PrerollSec;
    Input.ObservedAudioBufferEndSec = Detector.GetObservedAudioBufferEndSec();
    Input.bInputStreamClosed = bInputStreamClosed;
    Input.NPCID = NPCID;
    Input.LineID = LineID;

    FOffgridAILipsyncRuntimeAdapter::UpdateCommittedTrack(Input, CommittedTrack, PunctuationHoldState, bCommittedTrackBuilt);
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
    Input.GapCandidates = &Detector.GetGapCandidates();
    Input.SoftLullCandidates = &Detector.GetSoftLullCandidates();
    Input.AudioFeatureFrames = &Detector.GetFeatureFrames();
    Input.CurrentPlaybackSec = PlaybackSec;
    Input.PrerollSec = PrerollSec;
    Input.ObservedAudioBufferEndSec = Detector.GetObservedAudioBufferEndSec();
    Input.bInputStreamClosed = true;
    Input.bPlaybackFinalized = true;
    Input.NPCID = NPCID;
    Input.LineID = LineID;

    FOffgridAILipsyncRuntimeAdapter::UpdateCommittedTrack(Input, CommittedTrack, PunctuationHoldState, bCommittedTrackBuilt);
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

    RuntimeCommitDiagnosticRows.Reset();
    RuntimeSpeechRegionDiagnosticRows.Reset();
    RuntimeBoundaryDiagnosticRows.Reset();
    for (const FOffgridAICommittedVisemeEvent& E : CommittedTrack.Events)
    {
        FOffgridAIRuntimeCommitDiagnosticRow R;
        R.LineID = LineID;
        R.UpdateOrdinal = RuntimeCommitDiagnosticUpdateOrdinal;
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
        R.bMappedToObservedSpeech = E.bMappedToObservedSpeech;
        R.bUsedInitialSpeechAnchor = E.bUsedInitialSpeechAnchor;
        R.bUsedResumeAnchor = E.bUsedResumeAnchor;
        R.AcousticAnchorKind = E.AcousticAnchorKind;
        R.AcousticAnchorSec = E.AcousticAnchorSeconds;
        R.AcousticAnchorErrorSec = E.AcousticAnchorErrorSeconds;
        R.ObservedPauseDecaySec = E.ObservedPauseDecaySeconds;
        R.ObservedResumeOnsetSec = E.ObservedResumeOnsetSeconds;
        R.ObservedResumeEnergyAnchorSec = E.ObservedResumeEnergyAnchorSeconds;
        R.BoundaryWordIndex = E.BoundaryWordIndex;
        R.BoundaryMark = E.BoundaryMark;
        R.BoundaryOutcome = E.BoundaryOutcome;
        R.DiagnosticKind = E.CommitReason;
        RuntimeCommitDiagnosticRows.Add(R);
    }

    for (const FOffgridAIStreamingSpeechRegion& SpeechRegion : ResolvedSpeechRegions)
    {
        FOffgridAIRuntimeSpeechRegionDiagnosticRow RegionRow;
        RegionRow.LineID = LineID;
        RegionRow.UpdateOrdinal = RuntimeCommitDiagnosticUpdateOrdinal;
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

        for (const FOffgridAICommittedVisemeEvent& E : CommittedTrack.Events)
        {
            if (E.SpeechRegionIndex == SpeechRegion.SpeechRegionIndex)
            {
                ++RegionRow.CommittedEventCount;
            }
        }


        RuntimeSpeechRegionDiagnosticRows.Add(RegionRow);
    }

    {
        FOffgridAIRuntimeBoundaryDiagnosticRow BoundaryRow;
        BoundaryRow.LineID = LineID;
        BoundaryRow.UpdateOrdinal = RuntimeCommitDiagnosticUpdateOrdinal;
        BoundaryRow.bFinalReplay = bFinalReplay;
        BoundaryRow.CurrentPlaybackSec = CurrentPlaybackSec;
        BoundaryRow.bPlayheadStarted = PunctuationHoldState.bPlayheadStarted;
        BoundaryRow.bHoldActive = PunctuationHoldState.bHoldActive;
        BoundaryRow.bObservedPauseLull = PunctuationHoldState.bObservedPauseLull;
        BoundaryRow.bSawConfirmedOutOfSpeechAfterBoundary = PunctuationHoldState.bSawConfirmedOutOfSpeechAfterBoundary;
        BoundaryRow.bResumeAnchorActive = PunctuationHoldState.bResumeAnchorActive;
        BoundaryRow.bResumeAnchorFromInitialSpeech = PunctuationHoldState.bResumeAnchorFromInitialSpeech;
        BoundaryRow.BoundaryWordIndex = PunctuationHoldState.BoundaryWordIndex;
        BoundaryRow.BoundaryMark = PunctuationHoldState.ActiveBoundaryMark != TCHAR(0)
            ? FString::Chr(PunctuationHoldState.ActiveBoundaryMark)
            : FString();
        BoundaryRow.PauseClass = BoundaryPauseClassToString(PunctuationHoldState.ActivePauseClass);
        BoundaryRow.HoldStartSpeechRegionIndex = PunctuationHoldState.HoldStartSpeechRegionIndex;
        BoundaryRow.ResumeAnchorEventIndex = PunctuationHoldState.ResumeAnchorEventIndex;
        BoundaryRow.PlaybackOriginSec = PunctuationHoldState.PlaybackOriginSec;
        BoundaryRow.LastPlaybackSec = PunctuationHoldState.LastPlaybackSec;
        BoundaryRow.ActivePlayheadSec = PunctuationHoldState.ActivePlayheadSec;
        BoundaryRow.TotalPausedSec = PunctuationHoldState.TotalPausedSec;
        BoundaryRow.HoldStartPlaybackSec = PunctuationHoldState.HoldStartPlaybackSec;
        BoundaryRow.BoundarySearchStartPlaybackSec = PunctuationHoldState.BoundarySearchStartPlaybackSec;
        BoundaryRow.HoldDeadlinePlaybackSec = PunctuationHoldState.HoldDeadlinePlaybackSec;
        BoundaryRow.HoldResumeTargetActiveSec = PunctuationHoldState.HoldResumeTargetActiveSec;
        BoundaryRow.HoldResumeTargetLeadSec = PunctuationHoldState.HoldResumeTargetLeadSec;
        BoundaryRow.ConfirmedQuietStartPlaybackSec = PunctuationHoldState.ConfirmedQuietStartPlaybackSec;
        BoundaryRow.QuietRMSNormAtDecay = PunctuationHoldState.QuietRMSNormAtDecay;
        BoundaryRow.QuietEvidenceAtDecay = PunctuationHoldState.QuietEvidenceAtDecay;
        BoundaryRow.QuietRawRMSAtDecay = PunctuationHoldState.QuietRawRMSAtDecay;
        BoundaryRow.ObservedResumeOnsetPlaybackSec = PunctuationHoldState.ObservedResumeOnsetPlaybackSec;
        BoundaryRow.ObservedResumeEnergyAnchorSec = PunctuationHoldState.ObservedResumeEnergyAnchorSec;
        BoundaryRow.ResumeAnchorActiveSec = PunctuationHoldState.ResumeAnchorActiveSec;
        BoundaryRow.ResumeAnchorLeadSec = PunctuationHoldState.ResumeAnchorLeadSec;
        BoundaryRow.ResumeAnchorFinalCenterSec = PunctuationHoldState.ResumeAnchorFinalCenterSec;
        BoundaryRow.LastFenceEstimatorOutcome = PunctuationHoldState.LastFenceEstimatorOutcome.ToString();
        BoundaryRow.LastResolvedBoundaryOutcome = PunctuationHoldState.LastResolvedBoundary.Outcome.ToString();
        BoundaryRow.LastResolvedBoundaryWordIndex = PunctuationHoldState.LastResolvedBoundary.WordIndex;
        BoundaryRow.LastResolvedBoundaryMark = PunctuationHoldState.LastResolvedBoundary.Mark != TCHAR(0)
            ? FString::Chr(PunctuationHoldState.LastResolvedBoundary.Mark)
            : FString();
        BoundaryRow.LastResolvedBoundaryDecaySec = PunctuationHoldState.LastResolvedBoundary.DecaySec;
        BoundaryRow.LastResolvedBoundaryResumeOnsetSec = PunctuationHoldState.LastResolvedBoundary.ResumeOnsetSec;
        BoundaryRow.LastResolvedBoundaryResumeEnergyAnchorSec = PunctuationHoldState.LastResolvedBoundary.ResumeEnergyAnchorSec;
        BoundaryRow.CommittedEventCount = CommittedTrack.Events.Num();
        BoundaryRow.DiagnosticKind = FName(TEXT("runtime_boundary_state"));
        RuntimeBoundaryDiagnosticRows.Add(BoundaryRow);
    }
    ++RuntimeCommitDiagnosticUpdateOrdinal;
}

void FOffgridAILipsyncRuntimeAdapter::UpdateCommittedTrack(const FOffgridAILipsyncRuntimeUpdateInput& Input, FOffgridAICommittedVisemeTrack& InOutTrack, FOffgridAIBoundaryPlaybackState& InOutHoldState, bool& bInOutTrackBuilt)
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

    const bool bStreamSealed = Input.bInputStreamClosed || Input.bPlaybackFinalized;
    const bool bPlaybackFinal = Input.bPlaybackFinalized;
    const float ObservedEnd = FMath::Max(Input.ObservedAudioBufferEndSec, 0.0f);

    TArray<FEffectiveSpeechRegion> EffectiveRegions;
    BuildEffectiveSpeechRegions(Input.SpeechRegions, ObservedEnd, bStreamSealed, EffectiveRegions);

    const float ObservedActiveSec = ComputeObservedActiveSpeechSeconds(EffectiveRegions);
    const float FirstSpeechStart = ComputeFirstSpeechStart(EffectiveRegions);
    const float LastSpeechEnd = ComputeLastSpeechEnd(EffectiveRegions, ObservedEnd);

    if (ObservedActiveSec <= 0.001f && !bStreamSealed)
    {
        return;
    }

    InOutTrack.SpeechStartSeconds = FirstSpeechStart >= 0.0f ? FirstSpeechStart : 0.0f;
    InOutTrack.SpeechEndSeconds = LastSpeechEnd;

    AdvancePlaybackHoldState(Input, EffectiveRegions, InOutHoldState);

    const float PlaybackOffsetSec =
        InOutHoldState.PlaybackOriginSec + InOutHoldState.TotalPausedSec;

    // Baseline pacing: within a continuous speech segment, use the text/CMU
    // duration priors exactly.  Audio only gates start/resume at punctuation; it
    // does not stretch, warp, or re-time intraline playback.


    // Runtime scheduling:
    // 1. transcript owns viseme identity and order,
    // 2. text/CMU priors own intraline pacing,
    // 3. acoustic start/resume anchors own only segment starts,
    // 4. punctuation pauses future commits only: finish the pre-boundary prefix,
    //    wait for decay/resume, then start the next text segment from the acoustic
    //    resume anchor.
    const float CommitLagSec = bPlaybackFinal ? 0.0f : 0.030f;
    const float MinLiveLeadSec = bPlaybackFinal ? 0.0f : 0.040f;
    const float RequestedLiveLeadSec = FMath::Max(Input.PrerollSec + 0.120f, 0.250f);
    const float MaxLiveLeadSec = bPlaybackFinal
        ? 999.0f
        : FMath::Min(RequestedLiveLeadSec, MaxLiveCommitLeadSec);

    int32 NextEventIndex = ComputeNextPlanEventIndex(InOutTrack);
    float LastCenter = ComputeLastCommittedCenterSec(InOutTrack);
    const float TotalPlannedActiveSec = FMath::Max(TotalPhoneActiveSeconds, 0.001f);

    float CommitSafeActiveSec = FMath::Max(InOutHoldState.ActivePlayheadSec - CommitLagSec, 0.0f);

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

        const float OrderNorm = EventCenterNorms.IsValidIndex(NextEventIndex)
            ? EventCenterNorms[NextEventIndex]
            : 1.0f;
        const int32 SourcePhoneGlobalIndex = T.SourcePhoneGlobalIndex;
        const float RequiredActiveSec = PhoneCenterActiveSeconds.IsValidIndex(SourcePhoneGlobalIndex)
            ? FMath::Max(PhoneCenterActiveSeconds[SourcePhoneGlobalIndex], 0.0f)
            : OrderNorm * TotalPlannedActiveSec;

        if (NextEventIndex == 0 && !InOutHoldState.bInitialSpeechAnchorResolved)
        {
            float InitialAnchorSec = -1.0f;
            if (FindInitialSpeechEnergyAnchorSec(Input.AudioFeatureFrames, EffectiveRegions, InitialAnchorSec))
            {
                InOutHoldState.bInitialSpeechAnchorResolved = true;
                InOutHoldState.bResumeAnchorActive = true;
                InOutHoldState.bResumeAnchorFromInitialSpeech = true;
                InOutHoldState.ResumeAnchorEventIndex = 0;
                InOutHoldState.ResumeAnchorActiveSec = RequiredActiveSec;
                InOutHoldState.ResumeAnchorLeadSec = LeadForPose(T.PoseID);
                InOutHoldState.ResumeAnchorFinalCenterSec = InitialAnchorSec;
                InOutHoldState.LastResolvedBoundary.DecaySec = -1.0f;
                InOutHoldState.LastResolvedBoundary.ResumeOnsetSec = InitialAnchorSec;
                InOutHoldState.LastResolvedBoundary.ResumeEnergyAnchorSec = InitialAnchorSec;
            }
            else if (!bStreamSealed)
            {
                break;
            }
            else
            {
                InOutHoldState.bInitialSpeechAnchorResolved = true;
            }
        }

        // A punctuation boundary must stop the *first* event that belongs to the
        // following word.  Do not rely on WordVisibleEventBeginIndices here: some
        // words have early glide/shape events before the planner's visible-begin
        // index, which allowed post-punctuation animation to leak through before
        // the hold was even created.
        if (T.WordIndex > 0 && !InOutHoldState.bHoldActive)
        {
            const int32 BoundaryWordIndex = T.WordIndex - 1;
            const TCHAR Boundary = Plan.WordBoundaryPunctuationAfter.IsValidIndex(BoundaryWordIndex)
                ? Plan.WordBoundaryPunctuationAfter[BoundaryWordIndex]
                : TCHAR(0);
            const EOffgridAIBoundaryPauseClass BoundaryPauseClass =
                Plan.WordBoundaryPauseClassAfter.IsValidIndex(BoundaryWordIndex)
                    ? Plan.WordBoundaryPauseClassAfter[BoundaryWordIndex]
                    : EOffgridAIBoundaryPauseClass::None;
            const EOffgridAIBoundaryPauseClass EffectivePauseClass =
                BoundaryPauseClass != EOffgridAIBoundaryPauseClass::None
                    ? BoundaryPauseClass
                    : (IsHardPausePunctuation(Boundary)
                        ? EOffgridAIBoundaryPauseClass::HardBreakPause
                        : (Boundary == TEXT(',')
                            ? EOffgridAIBoundaryPauseClass::SoftListPause
                            : EOffgridAIBoundaryPauseClass::None));
            const float HoldSeconds = HoldSecondsForBoundaryClass(Boundary, EffectivePauseClass);
            if (HoldSeconds > 0.0f && InOutHoldState.BoundaryWordIndex != BoundaryWordIndex)
            {
                InOutHoldState.bHoldActive = true;
                InOutHoldState.bObservedPauseLull = false;
                InOutHoldState.bResumeAnchorActive = false;
                InOutHoldState.bResumeAnchorFromInitialSpeech = false;
                InOutHoldState.ResumeAnchorEventIndex = NextEventIndex;
                InOutHoldState.BoundaryWordIndex = BoundaryWordIndex;
                InOutHoldState.ActivePauseClass = EffectivePauseClass;
                InOutHoldState.HoldStartPlaybackSec = Input.CurrentPlaybackSec;
                const float LastCommittedCenterSec = InOutTrack.Events.Num() > 0
                    ? InOutTrack.Events.Last().FinalRenderCenterSeconds
                    : Input.CurrentPlaybackSec;
                // Search a small amount before the visual boundary. Real acoustic valleys often
                // begin during the final consonant/decay of the preceding word. This changes only
                // where decay evidence is searched; punctuation patience still starts now.
                InOutHoldState.BoundarySearchStartPlaybackSec = FMath::Max(
                    FMath::Min(Input.CurrentPlaybackSec, LastCommittedCenterSec) - 0.080f,
                    0.0f);
                InOutHoldState.HoldDeadlinePlaybackSec = Input.CurrentPlaybackSec + HoldSeconds;
                InOutHoldState.HoldStartSpeechRegionIndex = FindRegionIndexAtPlayback(EffectiveRegions, Input.CurrentPlaybackSec);
                InOutHoldState.bSawConfirmedOutOfSpeechAfterBoundary = false;
                InOutHoldState.ConfirmedQuietStartPlaybackSec = -1.0f;
                InOutHoldState.QuietRMSNormAtDecay = 1.0f;
                InOutHoldState.QuietEvidenceAtDecay = 1.0f;
                InOutHoldState.QuietRawRMSAtDecay = 1.0f;
                InOutHoldState.ActiveBoundaryMark = Boundary;
                InOutHoldState.LastResolvedBoundary.Outcome = NAME_None;
                InOutHoldState.LastResolvedBoundary.WordIndex = INDEX_NONE;
                InOutHoldState.LastResolvedBoundary.Mark = TCHAR(0);
                InOutHoldState.ObservedResumeOnsetPlaybackSec = -1.0f;
                InOutHoldState.ObservedResumeEnergyAnchorSec = -1.0f;
                InOutHoldState.LastResolvedBoundary.DecaySec = -1.0f;
                InOutHoldState.LastResolvedBoundary.ResumeOnsetSec = -1.0f;
                InOutHoldState.LastResolvedBoundary.ResumeEnergyAnchorSec = -1.0f;
                InOutHoldState.HoldResumeTargetActiveSec = RequiredActiveSec;
                InOutHoldState.HoldResumeTargetLeadSec = LeadForPose(T.PoseID);
                InOutHoldState.HoldPreviousCommittedRenderEndSec = (InOutTrack.Events.Num() == 0)
                    ? -1.0f
                    : InOutTrack.Events.Last().RenderEndSeconds;
                if (bPlaybackFinal)
                {
                    // Final drain may encounter a later punctuation boundary only
                    // after earlier events have just been committed in this same
                    // pass. Resolve the newly opened hold immediately so the
                    // remaining suffix is not stranded behind a break with no
                    // subsequent update tick to release it.
                    AdvancePlaybackHoldState(Input, EffectiveRegions, InOutHoldState);
                    if (!InOutHoldState.bHoldActive)
                    {
                        continue;
                    }
                }

                break;
            }
        }

        // Boundary invariant: the first event after punctuation may not silently
        // fall through to normal playhead timing. It must either be released by
        // a confirmed resume anchor or by an explicit no-decay/final outcome.
        if (T.WordIndex > 0)
        {
            const int32 EventBoundaryWordIndex = T.WordIndex - 1;
            const TCHAR EventBoundary = Plan.WordBoundaryPunctuationAfter.IsValidIndex(EventBoundaryWordIndex)
                ? Plan.WordBoundaryPunctuationAfter[EventBoundaryWordIndex]
                : TCHAR(0);
            const EOffgridAIBoundaryPauseClass EventBoundaryClass = Plan.WordBoundaryPauseClassAfter.IsValidIndex(EventBoundaryWordIndex)
                ? Plan.WordBoundaryPauseClassAfter[EventBoundaryWordIndex]
                : EOffgridAIBoundaryPauseClass::None;
            if (HoldSecondsForBoundaryClass(EventBoundary, EventBoundaryClass) > 0.0f
                && InOutHoldState.BoundaryWordIndex == EventBoundaryWordIndex
                && InOutHoldState.LastResolvedBoundary.WordIndex != EventBoundaryWordIndex
                && !InOutHoldState.bResumeAnchorActive)
            {
                break;
            }
        }

        // Do not let CloseInputStream turn live playback into a suffix dump.
        // Buffered audio may still be audibly draining after input is sealed,
        // so the text-prior frontier remains bounded until Finalize().
        if (RequiredActiveSec > CommitSafeActiveSec && !bPlaybackFinal)
        {
            break;
        }

        const float RequiredPhoneStartActiveSec = PhoneStartActiveSeconds.IsValidIndex(SourcePhoneGlobalIndex)
            ? FMath::Max(PhoneStartActiveSeconds[SourcePhoneGlobalIndex], 0.0f)
            : FMath::Max(RequiredActiveSec - 0.040f, 0.0f);
        const float RequiredPhoneEndActiveSec = PhoneEndActiveSeconds.IsValidIndex(SourcePhoneGlobalIndex)
            ? FMath::Max(PhoneEndActiveSeconds[SourcePhoneGlobalIndex], 0.0f)
            : (RequiredActiveSec + 0.040f);

        const float PoseLeadSec = LeadForPose(T.PoseID);
        float BaseStart = PlaybackOffsetSec + RequiredPhoneStartActiveSec;
        float Center = PlaybackOffsetSec + RequiredActiveSec;
        float BaseEnd = PlaybackOffsetSec + RequiredPhoneEndActiveSec;
        bool bUsedResumeAnchor = false;
        bool bUsedInitialSpeechAnchor = false;
        bool bUsingResumeAnchorSchedule = false;
        FName AcousticAnchorKind = NAME_None;
        float AcousticAnchorSec = -1.0f;
        int32 EventBoundaryWordIndex = INDEX_NONE;
        TCHAR EventBoundaryMark = TCHAR(0);
        EOffgridAIBoundaryPauseClass EventBoundaryPauseClass = EOffgridAIBoundaryPauseClass::None;
        FName EventBoundaryOutcome = NAME_None;
        if (T.WordIndex > 0)
        {
            EventBoundaryWordIndex = T.WordIndex - 1;
            EventBoundaryMark = Plan.WordBoundaryPunctuationAfter.IsValidIndex(EventBoundaryWordIndex)
                ? Plan.WordBoundaryPunctuationAfter[EventBoundaryWordIndex]
                : TCHAR(0);
            EventBoundaryPauseClass = Plan.WordBoundaryPauseClassAfter.IsValidIndex(EventBoundaryWordIndex)
                ? Plan.WordBoundaryPauseClassAfter[EventBoundaryWordIndex]
                : EOffgridAIBoundaryPauseClass::None;
            if (HoldSecondsForBoundaryClass(EventBoundaryMark, EventBoundaryPauseClass) <= 0.0f)
            {
                EventBoundaryWordIndex = INDEX_NONE;
                EventBoundaryMark = TCHAR(0);
            }
            else if (InOutHoldState.LastResolvedBoundary.WordIndex == EventBoundaryWordIndex)
            {
                EventBoundaryOutcome = InOutHoldState.LastResolvedBoundary.Outcome;
            }
        }

        const bool bScheduleUsesResumeAnchor = InOutHoldState.bResumeAnchorActive
            && NextEventIndex >= InOutHoldState.ResumeAnchorEventIndex;
        const bool bExactAcousticAnchorEvent = bScheduleUsesResumeAnchor
            && NextEventIndex == InOutHoldState.ResumeAnchorEventIndex;

        if (bScheduleUsesResumeAnchor)
        {
            bUsingResumeAnchorSchedule = true;
            // The target active time captured when the hold was created can go
            // stale while streaming regions grow.  The invariant we actually
            // need is simpler: the first post-boundary event center equals the
            // acoustic anchor.  When that event is reached, refresh the schedule
            // origin from the event's current active time and lead; later events
            // then inherit timing relative to the now-correct anchor.
            if (bExactAcousticAnchorEvent)
            {
                InOutHoldState.ResumeAnchorActiveSec = RequiredActiveSec;
                InOutHoldState.ResumeAnchorLeadSec = PoseLeadSec;
            }

            // Schedule the resumed segment from the first post-punctuation phone,
            // not from a globally delayed clock. This preserves the text-duration
            // prior inside the resumed word/sentence while putting the visible
            // restart exactly on the observed audio restart.
            const float LeadDeltaSec = PoseLeadSec - InOutHoldState.ResumeAnchorLeadSec;
            BaseStart = InOutHoldState.ResumeAnchorFinalCenterSec
                + (RequiredPhoneStartActiveSec - InOutHoldState.ResumeAnchorActiveSec)
                - LeadDeltaSec;
            Center = InOutHoldState.ResumeAnchorFinalCenterSec
                + (RequiredActiveSec - InOutHoldState.ResumeAnchorActiveSec)
                - LeadDeltaSec;
            BaseEnd = InOutHoldState.ResumeAnchorFinalCenterSec
                + (RequiredPhoneEndActiveSec - InOutHoldState.ResumeAnchorActiveSec)
                - LeadDeltaSec;

            // Only the actual anchored event should carry anchor diagnostics.
            // Later events are scheduled relative to that anchor but are not
            // themselves anchor errors.
            if (bExactAcousticAnchorEvent)
            {
                bUsedResumeAnchor = true;
                bUsedInitialSpeechAnchor = InOutHoldState.bResumeAnchorFromInitialSpeech;
                AcousticAnchorKind = bUsedInitialSpeechAnchor
                    ? FName(TEXT("initial_energy_anchor"))
                    : FName(TEXT("punctuation_resume_energy_anchor"));
                AcousticAnchorSec = (!bUsedInitialSpeechAnchor && InOutHoldState.LastResolvedBoundary.ResumeEnergyAnchorSec >= 0.0f)
                    ? InOutHoldState.LastResolvedBoundary.ResumeEnergyAnchorSec
                    : InOutHoldState.ResumeAnchorFinalCenterSec;
            }
        }

        const float PriorCenter = Center;

        const float BaseSpan = FMath::Max(BaseEnd - BaseStart, 0.020f);
        const float Span = FMath::Max(SpanForPose(T.PoseID), BaseSpan);
        if (!bUsingResumeAnchorSchedule)
        {
            Center = FMath::Max(Center - PoseLeadSec, 0.0f);
        }
        const float LeadAdjustedCenter = Center;
        float MinLiveLeadDelay = 0.0f;
        float InterEventFloorDelay = 0.0f;

        if (RequiredActiveSec <= 0.001f)
        {
            Center = FMath::Max(Center, PlaybackOffsetSec + FMath::Min(Span * 0.42f, 0.060f));
        }
        if (!bPlaybackFinal && !bExactAcousticAnchorEvent)
        {
            const float MinAllowedCenter = Input.CurrentPlaybackSec + MinLiveLeadSec;
            if (Center < MinAllowedCenter)
            {
                MinLiveLeadDelay = MinAllowedCenter - Center;
                Center = MinAllowedCenter;
            }
        }
        if (LastCenter >= 0.0f && !bExactAcousticAnchorEvent)
        {
            const float MinSpacingCenter = LastCenter + 0.050f;
            if (Center < MinSpacingCenter)
            {
                InterEventFloorDelay = MinSpacingCenter - Center;
                Center = MinSpacingCenter;
            }
        }
        if (!bPlaybackFinal)
        {
            const float CommitLeadSec = Center - Input.CurrentPlaybackSec;
            if (CommitLeadSec > MaxLiveLeadSec)
            {
                break;
            }

            // Very late live commits are not useful: by the time the event is
            // materialized, the renderer has already passed its visible window.
            // Breaking instead of dumping preserves the existing live cadence and
            // lets the next tick/final drain make an explicit decision.
            if (CommitLeadSec < -MaxLiveCommitBehindSec)
            {
                break;
            }
        }

        FName EffectiveCommitReason = TextPriorMonotonicCommitReason;
        if (bExactAcousticAnchorEvent)
        {
            EffectiveCommitReason = InOutHoldState.bResumeAnchorFromInitialSpeech
                ? FName(TEXT("initial_anchor_commit"))
                : FName(TEXT("punctuation_resume_anchor_commit"));
            if (!InOutHoldState.bResumeAnchorFromInitialSpeech)
            {
                EventBoundaryOutcome = FName(TEXT("confirmed_resume_anchor"));
            }
        }
        else if (EventBoundaryWordIndex != INDEX_NONE
            && InOutHoldState.LastResolvedBoundary.WordIndex == EventBoundaryWordIndex
            && InOutHoldState.LastResolvedBoundary.Outcome == FName(TEXT("no_decay_continuous")))
        {
            EffectiveCommitReason = FName(TEXT("boundary_no_decay_continuous_commit"));
            EventBoundaryOutcome = InOutHoldState.LastResolvedBoundary.Outcome;
        }

        FOffgridAICommittedVisemeEvent E;
        FillEventFromText(
            Plan,
            NextEventIndex,
            T,
            OrderNorm,
            Center,
            Span,
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
        E.SpeechRegionIndex = FMath::Max(FindRegionIndexAtPlayback(EffectiveRegions, Input.CurrentPlaybackSec), 0);
        E.CommitReason = EffectiveCommitReason;
        E.bUsedInitialSpeechAnchor = bUsedInitialSpeechAnchor;
        E.bUsedResumeAnchor = bUsedResumeAnchor && !bUsedInitialSpeechAnchor;
        E.AcousticAnchorKind = AcousticAnchorKind;
        E.AcousticAnchorSeconds = AcousticAnchorSec;
        E.AcousticAnchorErrorSeconds = AcousticAnchorSec >= 0.0f
            ? E.FinalRenderCenterSeconds - AcousticAnchorSec
            : 0.0f;
        E.ObservedPauseDecaySeconds = InOutHoldState.LastResolvedBoundary.DecaySec;
        E.ObservedResumeOnsetSeconds = InOutHoldState.LastResolvedBoundary.ResumeOnsetSec;
        E.ObservedResumeEnergyAnchorSeconds = InOutHoldState.LastResolvedBoundary.ResumeEnergyAnchorSec;
        E.BoundaryWordIndex = EventBoundaryWordIndex;
        if (EventBoundaryMark != TCHAR(0))
        {
            E.BoundaryMark = FString::Chr(EventBoundaryMark);
        }
        E.BoundaryOutcome = EventBoundaryOutcome;

        auto ShiftEventForwardToCenter = [](FOffgridAICommittedVisemeEvent& Event, float ForcedCenter)
        {
            if (ForcedCenter <= Event.FinalRenderCenterSeconds)
            {
                return;
            }

            const float Delta = ForcedCenter - Event.FinalRenderCenterSeconds;
            Event.FinalRenderCenterSeconds += Delta;
            Event.RenderStartSeconds += Delta;
            Event.RenderEndSeconds += Delta;
        };

        if (EffectiveCommitReason == FName(TEXT("boundary_no_decay_continuous_commit"))
            && InOutTrack.Events.Num() > 0)
        {
            const FOffgridAICommittedVisemeEvent& PreviousCommittedEvent = InOutTrack.Events.Last();
            const bool bFirstPostBoundaryEvent =
                EventBoundaryWordIndex != INDEX_NONE
                && PreviousCommittedEvent.WordIndex <= EventBoundaryWordIndex;
            if (bFirstPostBoundaryEvent)
            {
                const float MinVisibleGapSec = MinVisibleGapForBoundaryClass(EventBoundaryMark, EventBoundaryPauseClass);
                const float MinRenderStartSec = PreviousCommittedEvent.RenderEndSeconds + MinVisibleGapSec;
                if (E.RenderStartSeconds < MinRenderStartSec)
                {
                    const float ForcedCenter = E.FinalRenderCenterSeconds + (MinRenderStartSec - E.RenderStartSeconds);
                    ShiftEventForwardToCenter(E, ForcedCenter);
                }
            }
        }

        // Timing is monotonic text-prior playback, reset only by acoustic start/resume anchors.
        if (bExactAcousticAnchorEvent && InOutTrack.Events.Num() > 0)
        {
            if (!InOutHoldState.bResumeAnchorFromInitialSpeech
                && InOutHoldState.LastResolvedBoundary.DecaySec >= 0.0f
                && EventBoundaryWordIndex != INDEX_NONE)
            {
                // When a punctuation pause/resume is confirmed after the text-prior
                // schedule has already drifted ahead, trailing pre-boundary events
                // may still be sitting in the future past the observed lull. Those
                // future events are visually wrong and would otherwise force the
                // resumed anchor behind them. Drop only the unplayed suffix that
                // belongs to the pre-boundary side and starts after the confirmed
                // quiet point; already-rendered history remains immutable.
                const float ConfirmedQuietStartSec = InOutHoldState.LastResolvedBoundary.DecaySec;
                while (InOutTrack.Events.Num() > 0)
                {
                    const FOffgridAICommittedVisemeEvent& TailEvent = InOutTrack.Events.Last();
                    if (TailEvent.WordIndex > EventBoundaryWordIndex)
                    {
                        break;
                    }
                    if (TailEvent.RenderStartSeconds < ConfirmedQuietStartSec)
                    {
                        break;
                    }
                    if (TailEvent.FinalRenderCenterSeconds <= Input.CurrentPlaybackSec + 0.001f)
                    {
                        break;
                    }
                    InOutTrack.Events.RemoveAt(InOutTrack.Events.Num() - 1, 1, EAllowShrinking::No);
                }

                LastCenter = InOutTrack.Events.Num() > 0
                    ? InOutTrack.Events.Last().FinalRenderCenterSeconds
                    : -1.0f;
            }

            if (InOutTrack.Events.Num() == 0)
            {
                LastCenter = -1.0f;
            }

            if (InOutTrack.Events.Num() > 0)
            {
            // Preserve the acoustic anchor when possible. Prefer trimming the
            // immediately previous visible tail to create the punctuation gap,
            // rather than pushing the resumed event later than the observed
            // restart.
                FOffgridAICommittedVisemeEvent& PreviousEvent = InOutTrack.Events.Last();
                const float MinVisibleGapSec = MinVisibleGapForBoundaryClass(EventBoundaryMark, EventBoundaryPauseClass);
                const float DesiredPreviousEnd = FMath::Max(
                    PreviousEvent.FinalRenderCenterSeconds,
                    E.RenderStartSeconds - MinVisibleGapSec);
                PreviousEvent.RenderEndSeconds = FMath::Min(PreviousEvent.RenderEndSeconds, DesiredPreviousEnd);

                const float MinRenderStartSec = PreviousEvent.RenderEndSeconds + MinVisibleGapSec;
                if (E.RenderStartSeconds < MinRenderStartSec)
                {
                    const float ForcedCenter = E.FinalRenderCenterSeconds + (MinRenderStartSec - E.RenderStartSeconds);
                    ShiftEventForwardToCenter(E, ForcedCenter);
                }
            }
        }
        if (LastCenter >= 0.0f && E.FinalRenderCenterSeconds < LastCenter + 0.001f)
        {
            float ForcedCenter = LastCenter + 0.001f;
            if (ForcedCenter < 0.0f)
            {
                ForcedCenter = 0.0f;
            }
            ShiftEventForwardToCenter(E, ForcedCenter);
        }
        if (E.AcousticAnchorSeconds >= 0.0f)
        {
            E.AcousticAnchorErrorSeconds = E.FinalRenderCenterSeconds - E.AcousticAnchorSeconds;
        }
        E.RenderStartSeconds = FMath::Min(E.RenderStartSeconds, E.FinalRenderCenterSeconds);
        E.RenderEndSeconds = FMath::Max(E.RenderEndSeconds, E.FinalRenderCenterSeconds);
        InOutTrack.Events.Add(E);
        LastCenter = E.FinalRenderCenterSeconds;
        ++NextEventIndex;
    }

    // The LineCoach samples only when the committed track is marked built.
    // In the current runtime, the track is intentionally live and prefix-built.
    // Once text-prior visemes are anchored by stable speech start/resume evidence
    // and committed, they are authoritative and must be available to the performer.
    // Waiting until final/full completion causes the whole front of each line to
    // be missed live, even though runtime_commit_events.csv contains early
    // committed centers.
    bInOutTrackBuilt = InOutTrack.Events.Num() > 0;
}
