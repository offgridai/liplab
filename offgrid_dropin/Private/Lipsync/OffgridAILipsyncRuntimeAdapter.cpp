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
        return 0.320f;
    }
    if (IsHardLikeBoundaryClass(C, PauseClass))
    {
        return 1.150f;
    }
    return 0.0f;
}

static float MinDecayToResumeGapForBoundaryClass(TCHAR C, EOffgridAIBoundaryPauseClass PauseClass)
{
    return IsHardLikeBoundaryClass(C, PauseClass) ? 0.120f : 0.020f;
}

static float MinVisibleGapForBoundaryClass(TCHAR C, EOffgridAIBoundaryPauseClass PauseClass)
{
    return IsHardLikeBoundaryClass(C, PauseClass) ? 0.140f : 0.035f;
}

static float MaxArtificialResumeDelayForBoundaryClass(TCHAR C, EOffgridAIBoundaryPauseClass PauseClass)
{
    return IsHardLikeBoundaryClass(C, PauseClass) ? 0.160f : 0.260f;
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
    const float ResumeBloomRawRMS = FMath::Max(0.004f, QuietRawRMSAtDecay * 3.0f + 0.001f);
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
            4,
            FMath::Max(0.004f, OutQuietRawRMS * 3.0f + 0.001f));
        if (IsStableResumeAnchorFrame(&Frame, OutQuietRMSNorm, OutQuietEvidence, OutQuietRawRMS)
            && bStableRawSpeechRun)
        {
            OutResumeOnsetSec = Frame.AudioBufferCenterSec;
            OutResumeAnchorSec = Frame.AudioBufferCenterSec;
            return true;
        }
    }

    return false;
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
        const int32 PlaybackFrameIndex = FindFeatureFrameIndexAtPlayback(Input.AudioFeatureFrames, PlaybackSec);
        const FOffgridAIStreamingAudioFeatureFrame* PlaybackFrame =
            (Input.AudioFeatureFrames && PlaybackFrameIndex != INDEX_NONE)
                ? &(*Input.AudioFeatureFrames)[PlaybackFrameIndex]
                : nullptr;

        const float MinPauseHoldSec = 0.050f;
        const float HoldElapsedSec = PlaybackSec - InOutState.HoldStartPlaybackSec;
        const bool bSpeechResumed = IsPunctuationPauseSpeechFrame(PlaybackFrame);

        // Decay is the boundary gate. It must represent a real post-punctuation
        // acoustic valley, not a single adaptive low-evidence frame. Without
        // this, comma/period holds can release inside the prior word tail and
        // appear to ignore transcript boundaries.
        const bool bObservedRawDecayThisFrame = PlaybackFrame
            && PlaybackFrameIndex != INDEX_NONE
            && HasStableRawQuietRunEndingAtFrame(
                Input.AudioFeatureFrames,
                PlaybackFrameIndex,
                3,
                InOutState.BoundarySearchStartPlaybackSec);
        const bool bObservedHardPauseDecayThisFrame = PlaybackFrame
            && PlaybackFrameIndex != INDEX_NONE
            && IsHardLikeBoundaryClass(InOutState.ActiveBoundaryMark, InOutState.ActivePauseClass)
            && HasStableHardPauseQuietRunEndingAtFrame(
                Input.AudioFeatureFrames,
                PlaybackFrameIndex,
                3,
                InOutState.BoundarySearchStartPlaybackSec);
        const bool bObservedDecayThisFrame = bObservedRawDecayThisFrame || bObservedHardPauseDecayThisFrame;
        if (bObservedDecayThisFrame)
        {
            InOutState.bObservedPauseLull = true;
            InOutState.bSawConfirmedOutOfSpeechAfterBoundary = true;

            const int32 QuietStartFrameIndex = FMath::Max(PlaybackFrameIndex - 2, 0);
            const FOffgridAIStreamingAudioFeatureFrame& QuietStartFrame = (*Input.AudioFeatureFrames)[QuietStartFrameIndex];
            if (InOutState.ConfirmedQuietStartPlaybackSec < 0.0f)
            {
                InOutState.ConfirmedQuietStartPlaybackSec = QuietStartFrame.AudioBufferCenterSec;
                InOutState.QuietRMSNormAtDecay = QuietStartFrame.RMSNorm;
                InOutState.QuietEvidenceAtDecay = QuietStartFrame.SpeechEvidence;
                InOutState.QuietRawRMSAtDecay = QuietStartFrame.RMS;
            }

            InOutState.QuietRMSNormAtDecay = FMath::Min(InOutState.QuietRMSNormAtDecay, PlaybackFrame->RMSNorm);
            InOutState.QuietEvidenceAtDecay = FMath::Min(InOutState.QuietEvidenceAtDecay, PlaybackFrame->SpeechEvidence);
            InOutState.QuietRawRMSAtDecay = FMath::Min(InOutState.QuietRawRMSAtDecay, PlaybackFrame->RMS);
        }

        const int32 ContainingRegionIndex = FindContainingRegionIndexAtPlayback(EffectiveRegions, PlaybackSec);
        const bool bEnteredLaterRegion = ContainingRegionIndex != INDEX_NONE
            && InOutState.HoldStartSpeechRegionIndex != INDEX_NONE
            && ContainingRegionIndex > InOutState.HoldStartSpeechRegionIndex;

        const float MinDecayToResumeGapSec = MinDecayToResumeGapForBoundaryClass(InOutState.ActiveBoundaryMark, InOutState.ActivePauseClass);
        const bool bResumePastMinimumHardPause = InOutState.ConfirmedQuietStartPlaybackSec < 0.0f
            || PlaybackFrame == nullptr
            || PlaybackFrame->AudioBufferCenterSec + 0.001f >= InOutState.ConfirmedQuietStartPlaybackSec + MinDecayToResumeGapSec;
        const bool bResumeAllowedByGapPolicy = bResumePastMinimumHardPause;
        const bool bStableRawSpeechRun = PlaybackFrame
            && PlaybackFrameIndex != INDEX_NONE
            && HasStableRawSpeechRunFromFrame(
                Input.AudioFeatureFrames,
                PlaybackFrameIndex,
                4,
                FMath::Max(0.004f, InOutState.QuietRawRMSAtDecay * 3.0f + 0.001f));
        const bool bStableResumeFrame = PlaybackFrame
            && IsStableResumeAnchorFrame(
                PlaybackFrame,
                InOutState.QuietRMSNormAtDecay,
                InOutState.QuietEvidenceAtDecay,
                InOutState.QuietRawRMSAtDecay)
            && bStableRawSpeechRun;

        const bool bFreshResumeOnset = InOutState.bObservedPauseLull
            && bSpeechResumed
            && (bEnteredLaterRegion || IsFreshResumeOnsetFrame(PlaybackFrame));
        const bool bHardResumeWithoutDetectorSplit = InOutState.bObservedPauseLull
            && IsHardLikeBoundaryClass(InOutState.ActiveBoundaryMark, InOutState.ActivePauseClass)
            && bSpeechResumed
            && bResumeAllowedByGapPolicy
            && bStableResumeFrame;
        if ((bFreshResumeOnset || bHardResumeWithoutDetectorSplit)
            && InOutState.ObservedResumeOnsetPlaybackSec < 0.0f)
        {
            InOutState.ObservedResumeOnsetPlaybackSec = PlaybackFrame ? PlaybackFrame->AudioBufferCenterSec : PlaybackSec;
        }

        const bool bHasConfirmedResumeOnset = InOutState.ObservedResumeOnsetPlaybackSec >= 0.0f;
        const bool bStableResumeAnchor = bHasConfirmedResumeOnset
            && PlaybackFrame
            && PlaybackFrameIndex != INDEX_NONE
            && PlaybackFrame->AudioBufferCenterSec + 0.001f >= InOutState.ObservedResumeOnsetPlaybackSec
            && bResumeAllowedByGapPolicy
            && bStableResumeFrame;

        bool bResolvedBufferedResumeThisTick = false;
        // Bounded live back-search: if playback has already reached a
        // boundary's quiet->resume transition but the per-frame detector missed
        // the exact transition tick, resolve from buffered frames only up to the
        // current playback window. This applies to commas and hard punctuation
        // alike; otherwise comma lulls can be missed live and then degrade to
        // finalized_abandoned/text-prior playback at finalization.
        if (!bHasConfirmedResumeOnset
            && !Input.bPlaybackFinalized
            && Input.AudioFeatureFrames
            && Input.ObservedAudioBufferEndSec + 0.001f >= InOutState.HoldStartPlaybackSec + MinPauseHoldSec)
        {
            float BufferedQuietSec = -1.0f;
            float BufferedResumeOnsetSec = -1.0f;
            float BufferedResumeAnchorSec = -1.0f;
            float BufferedQuietRMSNorm = 1.0f;
            float BufferedQuietEvidence = 1.0f;
            float BufferedQuietRawRMS = 1.0f;
            const float MaxLiveBoundarySearchSec = FMath::Min(
                Input.ObservedAudioBufferEndSec,
                PlaybackSec + 0.060f);
            const bool bFoundBufferedResume = FindBufferedBoundaryResumeAnchorSec(
                Input.AudioFeatureFrames,
                InOutState.ActiveBoundaryMark,
                InOutState.ActivePauseClass,
                InOutState.BoundarySearchStartPlaybackSec,
                MaxLiveBoundarySearchSec,
                MinDecayToResumeGapForBoundaryClass(InOutState.ActiveBoundaryMark, InOutState.ActivePauseClass),
                BufferedQuietSec,
                BufferedResumeOnsetSec,
                BufferedResumeAnchorSec,
                BufferedQuietRMSNorm,
                BufferedQuietEvidence,
                BufferedQuietRawRMS);

            if (bFoundBufferedResume
                && BufferedResumeAnchorSec <= PlaybackSec + 0.070f)
            {
                InOutState.bObservedPauseLull = true;
                InOutState.bSawConfirmedOutOfSpeechAfterBoundary = true;
                InOutState.ConfirmedQuietStartPlaybackSec = BufferedQuietSec;
                InOutState.ObservedResumeOnsetPlaybackSec = BufferedResumeOnsetSec;
                InOutState.QuietRMSNormAtDecay = BufferedQuietRMSNorm;
                InOutState.QuietEvidenceAtDecay = BufferedQuietEvidence;
                InOutState.QuietRawRMSAtDecay = BufferedQuietRawRMS;
                ReanchorPausedClockToAcousticAnchor(BufferedResumeAnchorSec);
                InOutState.LastResolvedBoundary.Outcome = FName(TEXT("confirmed_resume_anchor_live_backsearch"));
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
                bResolvedBufferedResumeThisTick = true;
            }
        }

        if (!bResolvedBufferedResumeThisTick
            && InOutState.bObservedPauseLull
            && HoldElapsedSec >= MinPauseHoldSec
            && bSpeechResumed
            && bHasConfirmedResumeOnset
            && bStableResumeAnchor)
        {
            const float AcousticAnchorSec = PlaybackFrame ? PlaybackFrame->AudioBufferCenterSec : PlaybackSec;
            ReanchorPausedClockToAcousticAnchor(AcousticAnchorSec);
            InOutState.LastResolvedBoundary.Outcome = FName(TEXT("confirmed_resume_anchor"));
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
        else if (!bResolvedBufferedResumeThisTick && Input.bPlaybackFinalized && HoldElapsedSec >= MinPauseHoldSec)
        {
            // Finalization means the full audio buffer is available. Before
            // abandoning a hard boundary, search the buffered frames for the
            // quiet/resume pair that may have arrived after the last live tick.
            // This preserves the simple invariant: hard punctuation either
            // anchors to an observed resume or has an explicit true fallback;
            // it should not silently release post-boundary events on text-prior
            // timing just because the final update arrived while the hold was
            // still active.
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
                InOutState.ConfirmedQuietStartPlaybackSec = BufferedQuietSec;
                InOutState.ObservedResumeOnsetPlaybackSec = BufferedResumeOnsetSec;
                InOutState.QuietRMSNormAtDecay = BufferedQuietRMSNorm;
                InOutState.QuietEvidenceAtDecay = BufferedQuietEvidence;
                InOutState.QuietRawRMSAtDecay = BufferedQuietRawRMS;
                ReanchorPausedClockToAcousticAnchor(BufferedResumeAnchorSec);
                InOutState.LastResolvedBoundary.Outcome = FName(TEXT("confirmed_resume_anchor_final_search"));
                InOutState.LastResolvedBoundary.WordIndex = InOutState.BoundaryWordIndex;
                InOutState.LastResolvedBoundary.Mark = InOutState.ActiveBoundaryMark;
            }
            else
            {
                // If final search did not find a quiet->resume pair and the
                // live gate never observed a valid decay, this boundary was
                // spoken through.  Do not mark it finalized_abandoned: that
                // outcome authorizes the post-punctuation word to fall through
                // on ordinary text-prior timing while looking like a failed
                // pause/resume.  Make the continuous outcome explicit for every
                // punctuation class.
                const bool bFinalNoDecayContinuous = !InOutState.bObservedPauseLull;
                if (bFinalNoDecayContinuous)
                {
                    InOutState.ActivePlayheadSec += FMath::Max(PlaybackSec - InOutState.HoldStartPlaybackSec, 0.0f);
                    bReleasedContinuousNoDecayThisTick = true;
                }
                InOutState.LastResolvedBoundary.Outcome = bFinalNoDecayContinuous
                    ? FName(TEXT("no_decay_continuous"))
                    : FName(TEXT("finalized_abandoned"));
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
            }

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
        else if (!bResolvedBufferedResumeThisTick
            && !InOutState.bObservedPauseLull
            && PlaybackSec >= InOutState.HoldDeadlinePlaybackSec
            && (Input.bPlaybackFinalized || Input.ObservedAudioBufferEndSec + 0.001f >= InOutState.HoldDeadlinePlaybackSec))
        {
            // Any punctuation may be spoken through.  If the patience window is
            // fully covered by audio and contains no real decay, release it as an
            // explicit continuous boundary instead of falling through silently or
            // waiting for finalization to produce finalized_abandoned.
            InOutState.ActivePlayheadSec += FMath::Max(PlaybackSec - InOutState.HoldStartPlaybackSec, 0.0f);
            InOutState.LastResolvedBoundary.Outcome = FName(TEXT("no_decay_continuous"));
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
        FName EventBoundaryOutcome = NAME_None;
        if (T.WordIndex > 0)
        {
            EventBoundaryWordIndex = T.WordIndex - 1;
            EventBoundaryMark = Plan.WordBoundaryPunctuationAfter.IsValidIndex(EventBoundaryWordIndex)
                ? Plan.WordBoundaryPunctuationAfter[EventBoundaryWordIndex]
                : TCHAR(0);
            const EOffgridAIBoundaryPauseClass EventBoundaryClass = Plan.WordBoundaryPauseClassAfter.IsValidIndex(EventBoundaryWordIndex)
                ? Plan.WordBoundaryPauseClassAfter[EventBoundaryWordIndex]
                : EOffgridAIBoundaryPauseClass::None;
            if (HoldSecondsForBoundaryClass(EventBoundaryMark, EventBoundaryClass) <= 0.0f)
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

        // Timing is monotonic text-prior playback, reset only by acoustic start/resume anchors.
        if (bExactAcousticAnchorEvent && InOutTrack.Events.Num() > 0)
        {
            // Preserve the acoustic anchor.  Only shorten the immediately previous
            // envelope tail if it overlaps the restart; never move old centers or
            // compact an entire pre-boundary word.
            FOffgridAICommittedVisemeEvent& PreviousEvent = InOutTrack.Events.Last();
            const float TrimmedPreviousEnd = FMath::Max(PreviousEvent.FinalRenderCenterSeconds, E.FinalRenderCenterSeconds - 0.001f);
            PreviousEvent.RenderEndSeconds = FMath::Min(PreviousEvent.RenderEndSeconds, TrimmedPreviousEnd);
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
