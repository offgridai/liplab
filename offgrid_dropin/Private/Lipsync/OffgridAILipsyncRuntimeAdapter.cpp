#include "Lipsync/OffgridAILipsyncRuntimeAdapter.h"
#include "Lipsync/OffgridAIAcousticEvidence.h"

namespace
{
static const FName TextPriorMonotonicCommitReason(TEXT("text_prior_monotonic_commit"));

// Phone identity remains transcript-owned. Acoustic evidence may only choose
// timing for the next monotonic transcript syllable candidates.
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

static bool IsSentenceTerminalPunctuation(TCHAR C)
{
    return C == TEXT('.') || C == TEXT('!') || C == TEXT('?');
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

static bool IsOxfordConjunctionBoundary(
    const FOffgridAITextVisemePlan* Plan,
    int32 BoundaryWordIndex)
{
    if (!Plan || !Plan->WordPhoneBeginIndices.IsValidIndex(BoundaryWordIndex + 1))
    {
        return false;
    }
    const int32 NextPhoneIndex = Plan->WordPhoneBeginIndices[BoundaryWordIndex + 1];
    if (!Plan->ExpectedPhones.IsValidIndex(NextPhoneIndex)) return false;
    const FString& NextWord = Plan->ExpectedPhones[NextPhoneIndex].SourceWord;
    // Planner source words are normalized to lowercase.
    return NextWord == TEXT("and") || NextWord == TEXT("or");
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
        // A clause comma can be spoken through. Waiting for the full sentence
        // fence window in that case creates a conspicuous stall and burst.
        if (C == TEXT(','))
        {
            return 0.520f;
        }
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
    return IsSoftListBoundaryClass(C, PauseClass) ? 0.200f : 0.060f;
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
    // consonant or speech-section tail; accepting that as punctuation decay lets the
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

static bool FindNextSpeechRegionResumeAnchorSec(
    const TArray<FOffgridAIStreamingSpeechRegion>* SpeechRegions,
    int32 HoldStartSpeechRegionIndex,
    float MinDecayToResumeGapSec,
    float& OutQuietStartSec,
    float& OutResumeOnsetSec,
    float& OutResumeAnchorSec)
{
    if (!SpeechRegions || SpeechRegions->Num() < 2)
    {
        return false;
    }

    for (int32 Index = 0; Index + 1 < SpeechRegions->Num(); ++Index)
    {
        const FOffgridAIStreamingSpeechRegion& Before = (*SpeechRegions)[Index];
        const FOffgridAIStreamingSpeechRegion& After = (*SpeechRegions)[Index + 1];
        if (!Before.bEnded || !After.bStarted)
        {
            continue;
        }
        if (Before.SpeechRegionIndex != HoldStartSpeechRegionIndex
            || After.SpeechRegionIndex <= HoldStartSpeechRegionIndex)
        {
            continue;
        }
        if (After.AudioBufferStartSec + 0.001f
            < Before.AudioBufferEndSec + MinDecayToResumeGapSec)
        {
            continue;
        }

        OutQuietStartSec = Before.AudioBufferEndSec;
        OutResumeOnsetSec = After.AudioBufferStartSec;
        OutResumeAnchorSec = After.AudioBufferStartSec;
        return true;
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

static bool IsCausalSyllablePulse(
    const TArray<FOffgridAIStreamingAudioFeatureFrame>& Frames,
    int32 FrameIndex,
    float& OutProminence);

static bool HasSufficientNucleusProgressBefore(
    const FOffgridAILipsyncRuntimeUpdateInput& Input,
    const FOffgridAIBoundaryPlaybackState& State,
    float CandidateCloseSec,
    float* OutLastPulseSec = nullptr,
    int32* OutObservedPulses = nullptr,
    int32* OutExpectedNuclei = nullptr)
{
    if (OutLastPulseSec) *OutLastPulseSec = -1.0f;
    if (OutObservedPulses) *OutObservedPulses = 0;
    if (OutExpectedNuclei) *OutExpectedNuclei = 0;
    if (!Input.TextPlan || !Input.AudioFeatureFrames)
    {
        return true;
    }
    const int32 FirstWordIndex = State.LastResolvedBoundary.WordIndex >= 0
        ? State.LastResolvedBoundary.WordIndex + 1
        : 0;
    int32 ExpectedNuclei = 0;
    for (const FOffgridAIExpectedPhone& Phone : Input.TextPlan->ExpectedPhones)
    {
        if (Phone.bIsVowel
            && Phone.WordIndex >= FirstWordIndex
            && Phone.WordIndex <= State.BoundaryWordIndex)
        {
            ++ExpectedNuclei;
        }
    }
    if (ExpectedNuclei <= 0)
    {
        return true;
    }
    if (OutExpectedNuclei) *OutExpectedNuclei = ExpectedNuclei;

    int32 ObservedPulses = 0;
    const TArray<FOffgridAIStreamingAudioFeatureFrame>& Frames = *Input.AudioFeatureFrames;
    for (int32 FrameIndex = 14; FrameIndex + 1 < Frames.Num(); ++FrameIndex)
    {
        const float FrameSec = Frames[FrameIndex].AudioBufferCenterSec;
        if (FrameSec + 0.001f < State.SyllableSectionStartAudioSec) continue;
        if (FrameSec > CandidateCloseSec + 0.001f) break;
        float Prominence = 0.0f;
        if (IsCausalSyllablePulse(Frames, FrameIndex, Prominence) && Prominence >= 0.10f)
        {
            ++ObservedPulses;
            if (OutLastPulseSec) *OutLastPulseSec = FrameSec;
        }
    }

    // The pulse detector has high precision but imperfect recall. Permit one
    // missed nucleus in a multi-syllable section while still distinguishing
    // the internal valley before a word's final pulse.
    // At a list fence, accepting one fewer pulse systematically advances
    // multi-syllable items after their first nucleus (tur-key, let-tuce,
    // to-ma-to). Require the transcript's complete nucleus count here.
    const int32 RequiredPulses = ExpectedNuclei;
    if (OutObservedPulses) *OutObservedPulses = ObservedPulses;
    return ObservedPulses >= RequiredPulses;
}

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

static void UpdateProvisionalListPause(
    const FOffgridAILipsyncRuntimeUpdateInput& Input,
    float PlaybackSec,
    FOffgridAIBoundaryPlaybackState& State)
{
    if (!State.bHoldActive
        || !IsSoftListBoundaryClass(State.ActiveBoundaryMark, State.ActivePauseClass)
        || !Input.SoftLullCandidates)
    {
        return;
    }

    const float MaxCandidateStartSec = FMath::Min(
        Input.ObservedAudioBufferEndSec,
        PlaybackSec + LiveResumeSearchLookaheadForBoundaryClass(State.ActiveBoundaryMark, State.ActivePauseClass));
    for (const FOffgridAIStreamingSoftLullCandidate& Lull : *Input.SoftLullCandidates)
    {
        if (Lull.FrameCount <= 0
            || Lull.LullDurationSec < 0.020f
            || Lull.LullStartSec + 0.020f < State.BoundarySearchStartPlaybackSec)
        {
            continue;
        }
        if (Lull.LullStartSec > MaxCandidateStartSec + 0.001f)
        {
            break;
        }

        const float Confidence = SoftLullConfidence(Lull);
        if (Confidence < 0.18f)
        {
            continue;
        }
        if (Lull.LullDurationSec < 0.100f
            && !HasSufficientNucleusProgressBefore(Input, State, Lull.LullStartSec))
        {
            continue;
        }

        const bool bSameCandidate = State.bProvisionalListPause
            && Lull.LullStartSec <= State.ProvisionalListPauseEndSec + 0.040f
            && Lull.LullEndSec + 0.040f >= State.ProvisionalListPauseStartSec;
        if (!State.bProvisionalListPause || bSameCandidate)
        {
            State.bProvisionalListPause = true;
            State.ProvisionalListPauseStartSec = State.ProvisionalListPauseStartSec < 0.0f
                ? Lull.LullStartSec
                : FMath::Min(State.ProvisionalListPauseStartSec, Lull.LullStartSec);
            State.ProvisionalListPauseEndSec = FMath::Max(State.ProvisionalListPauseEndSec, Lull.LullEndSec);
            State.ProvisionalListPauseConfidence = FMath::Max(State.ProvisionalListPauseConfidence, Confidence);
            State.ProvisionalListPauseRMSNorm = FMath::Min(State.ProvisionalListPauseRMSNorm, Lull.RelativeRMSMin);
            State.ProvisionalListPauseEvidence = FMath::Min(
                State.ProvisionalListPauseEvidence,
                Lull.EvidenceSum / Lull.FrameCount);
        }
    }
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

        const float EarliestBacktrackSec = FMath::Max(
            QuietStartSec + MinDecayToResumeGapForBoundaryClass(BoundaryMark, PauseClass),
            Frame.AudioBufferCenterSec - 0.080f);
        const float OnsetRawRMS = FMath::Max(0.0020f, QuietRawRMS * 1.5f + 0.0005f);
        const float OnsetEvidence = FMath::Max(QuietEvidence + 0.05f, 0.12f);
        int32 EarliestOnsetFrameIndex = FrameIndex;
        int32 ConsecutiveFloorFrames = 0;
        for (int32 BackIndex = FrameIndex - 1; BackIndex >= 0; --BackIndex)
        {
            const FOffgridAIStreamingAudioFeatureFrame& BackFrame = (*AudioFeatureFrames)[BackIndex];
            if (BackFrame.AudioBufferCenterSec + 0.001f < EarliestBacktrackSec)
            {
                break;
            }
            const bool bMeaningfulEnergy = BackFrame.RMS >= OnsetRawRMS
                || BackFrame.SpeechEvidence >= OnsetEvidence
                || BackFrame.bStrongOnsetAnchor;
            if (bMeaningfulEnergy)
            {
                EarliestOnsetFrameIndex = BackIndex;
                ConsecutiveFloorFrames = 0;
            }
            else if (++ConsecutiveFloorFrames >= 2)
            {
                break;
            }
        }

        const float ResumeOnsetSec = (*AudioFeatureFrames)[EarliestOnsetFrameIndex].AudioBufferCenterSec;
        OutResumeOnsetSec = ResumeOnsetSec;
        OutResumeAnchorSec = ResumeOnsetSec;
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
    const bool bSoftListBoundary = IsSoftListBoundaryClass(State.ActiveBoundaryMark, State.ActivePauseClass);
    FCausalBoundaryFenceEstimate BestSoftListEstimate;
    float BestSoftListScore = -1.0f;
    auto ConsiderSoftListEstimate = [&](const FCausalBoundaryFenceEstimate& Candidate, float AcousticConfidence)
    {
        const float Proximity = FMath::Clamp(
            1.0f - FMath::Abs(Candidate.CloseSec - PlaybackSec) / 0.200f,
            0.0f,
            1.0f);
        const float Score = AcousticConfidence * 0.75f
            + Proximity * 0.20f
            + (Candidate.bObservedResume ? 0.05f : 0.0f);
        if (Score > BestSoftListScore)
        {
            BestSoftListScore = Score;
            BestSoftListEstimate = Candidate;
        }
    };
    const float HoldWindowSec = HoldSecondsForBoundaryClass(State.ActiveBoundaryMark, State.ActivePauseClass);
    const float MaxObservedSearchSec = (Input.bInputStreamClosed || Input.bPlaybackFinalized)
        ? Input.ObservedAudioBufferEndSec
        : FMath::Min(
            Input.ObservedAudioBufferEndSec,
            PlaybackSec + LiveResumeSearchLookaheadForBoundaryClass(State.ActiveBoundaryMark, State.ActivePauseClass));

    const float MinResumeGapSec = MinDecayToResumeGapForBoundaryClass(State.ActiveBoundaryMark, State.ActivePauseClass);
    if (bSoftListBoundary && State.bProvisionalListPause)
    {
        const float CandidateDurationSec = FMath::Max(
            State.ProvisionalListPauseEndSec - State.ProvisionalListPauseStartSec,
            0.0f);
        const bool bMatureClose = State.ProvisionalListPauseConfidence >= 0.40f
            || CandidateDurationSec >= 0.070f;
        float ResumeOnsetSec = -1.0f;
        float ResumeAnchorSec = -1.0f;
        const bool bFoundResume = FindStableResumeAnchorAfterTime(
            Input.AudioFeatureFrames,
            State.ActiveBoundaryMark,
            State.ActivePauseClass,
            State.ProvisionalListPauseStartSec,
            State.ProvisionalListPauseRMSNorm,
            State.ProvisionalListPauseEvidence,
            0.001f,
            State.ProvisionalListPauseEndSec,
            MaxObservedSearchSec,
            ResumeOnsetSec,
            ResumeAnchorSec);
        if (bMatureClose && bFoundResume)
        {
            Estimate.bObservedClose = true;
            Estimate.bObservedResume = true;
            Estimate.bUsedSoftLull = true;
            Estimate.CloseSec = State.ProvisionalListPauseStartSec;
            Estimate.ResumeOnsetSec = ResumeOnsetSec;
            Estimate.ResumeAnchorSec = ResumeAnchorSec;
            Estimate.QuietRMSNorm = State.ProvisionalListPauseRMSNorm;
            Estimate.QuietEvidence = State.ProvisionalListPauseEvidence;
            Estimate.QuietRawRMS = 0.001f;
            Estimate.Outcome = FName(TEXT("confirmed_resume_anchor_provisional_list"));
            return Estimate;
        }
        if (!bMatureClose
            && bFoundResume
            && Input.ObservedAudioBufferEndSec >= State.ProvisionalListPauseEndSec + 0.080f)
        {
            Estimate.bContinuousSpeech = true;
            Estimate.Outcome = FName(TEXT("provisional_list_shallow_continuation"));
            return Estimate;
        }
        return Estimate;
    }
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

            FCausalBoundaryFenceEstimate Candidate;
            Candidate.bObservedClose = true;
            Candidate.CloseSec = Gap.GapStartSec + FMath::Min(0.060f, FMath::Max(0.020f, Gap.GapDurationSec * 0.20f));
            Candidate.QuietRMSNorm = Gap.QuietRMSNorm;
            Candidate.QuietEvidence = Gap.QuietEvidence;
            Candidate.QuietRawRMS = 0.001f;
            Candidate.Outcome = FName(TEXT("pause_close_candidate"));
            if (bSoftListBoundary
                && Gap.GapDurationSec < 0.100f
                && !HasSufficientNucleusProgressBefore(Input, State, Candidate.CloseSec))
            {
                continue;
            }

            if (ResumeConfidence >= MinResumeConfidenceForBoundaryClass(State.ActiveBoundaryMark, State.ActivePauseClass)
                && Gap.GapEndSec + 0.001f >= Candidate.CloseSec + MinResumeGapSec)
            {
                float ResumeOnsetSec = -1.0f;
                float ResumeAnchorSec = -1.0f;
                const float QuietRawRMS = FMath::Max(0.001f, Gap.GapRMSNormMin * 0.05f);
                if (FindStableResumeAnchorAfterTime(
                        Input.AudioFeatureFrames,
                        State.ActiveBoundaryMark,
                        State.ActivePauseClass,
                        Candidate.CloseSec,
                        Gap.QuietRMSNorm,
                        Gap.QuietEvidence,
                        QuietRawRMS,
                        Gap.GapEndSec,
                        MaxObservedSearchSec,
                        ResumeOnsetSec,
                        ResumeAnchorSec))
                {
                    Candidate.bObservedResume = true;
                    Candidate.ResumeOnsetSec = ResumeOnsetSec;
                    Candidate.ResumeAnchorSec = ResumeAnchorSec;
                    Candidate.QuietRawRMS = QuietRawRMS;
                    Candidate.Outcome = FName(TEXT("confirmed_resume_anchor_gap"));
                }
            }

            if (bSoftListBoundary)
            {
                ConsiderSoftListEstimate(Candidate, (CloseConfidence + ResumeConfidence) * 0.5f);
                continue;
            }

            Estimate = Candidate;
            if (Candidate.bObservedResume)
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

            FCausalBoundaryFenceEstimate Candidate;
            Candidate.bObservedClose = true;
            Candidate.bUsedSoftLull = true;
            Candidate.CloseSec = Lull.LullStartSec + FMath::Min(0.050f, Lull.LullDurationSec);
            Candidate.QuietRMSNorm = Lull.RelativeRMSMin;
            Candidate.QuietEvidence = Lull.FrameCount > 0 ? (Lull.EvidenceSum / Lull.FrameCount) : 0.0f;
            Candidate.QuietRawRMS = 0.001f;
            Candidate.Outcome = FName(TEXT("pause_close_soft_lull"));
            if (bSoftListBoundary
                && Lull.LullDurationSec < 0.100f
                && !HasSufficientNucleusProgressBefore(Input, State, Candidate.CloseSec))
            {
                continue;
            }

            float ResumeOnsetSec = -1.0f;
            float ResumeAnchorSec = -1.0f;
            if (FindStableResumeAnchorAfterTime(
                    Input.AudioFeatureFrames,
                    State.ActiveBoundaryMark,
                    State.ActivePauseClass,
                    Candidate.CloseSec,
                    Candidate.QuietRMSNorm,
                    Candidate.QuietEvidence,
                    Candidate.QuietRawRMS,
                    Lull.LullEndSec,
                    MaxObservedSearchSec,
                    ResumeOnsetSec,
                    ResumeAnchorSec))
            {
                Candidate.bObservedResume = true;
                Candidate.ResumeOnsetSec = ResumeOnsetSec;
                Candidate.ResumeAnchorSec = ResumeAnchorSec;
                Candidate.Outcome = FName(TEXT("confirmed_resume_anchor_soft_lull"));
            }

            if (bSoftListBoundary)
            {
                ConsiderSoftListEstimate(Candidate, Confidence);
                continue;
            }
            return Candidate;
        }
    }

    if (BestSoftListEstimate.bObservedClose)
    {
        return BestSoftListEstimate;
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
        else if (Input.bInputStreamClosed
            && !Input.bPlaybackFinalized
            && (State.ActiveBoundaryMark == TEXT(',')
                || (IsSentenceTerminalPunctuation(State.ActiveBoundaryMark)
                    && Estimate.CloseSec >= State.HoldDeadlinePlaybackSec + 0.300f)))
        {
            // With complete input, a close that has no later resume cannot own
            // an intraline punctuation fence: there are planned phones after
            // the fence, so this is end-of-line quiet after that suffix was
            // already spoken.
            Estimate.bObservedClose = false;
            Estimate.bContinuousSpeech = true;
            Estimate.CloseSec = -1.0f;
            Estimate.Outcome = FName(TEXT("input_closed_trailing_quiet_continuous"));
        }
        return Estimate;
    }

    if (bSoftListBoundary)
    {
        float LastPulseSec = -1.0f;
        int32 ObservedPulses = 0;
        int32 ExpectedNuclei = 0;
        const bool bNucleiComplete = HasSufficientNucleusProgressBefore(
            Input,
            State,
            MaxObservedSearchSec,
            &LastPulseSec,
            &ObservedPulses,
            &ExpectedNuclei);
        const bool bNextItemPulseObserved = bNucleiComplete
            && ExpectedNuclei > 0
            && ObservedPulses > ExpectedNuclei;
        if (bNextItemPulseObserved)
        {
            Estimate.bContinuousSpeech = true;
            Estimate.Outcome = FName(TEXT("list_next_item_pulse_continuous"));
            return Estimate;
        }
        const float SoftDecisionWindowSec = IsOxfordConjunctionBoundary(
            Input.TextPlan,
            State.BoundaryWordIndex)
                ? HoldWindowSec
                : 0.750f;
        if (!Input.bPlaybackFinalized
            && PlaybackSec < State.HoldStartPlaybackSec + SoftDecisionWindowSec)
        {
            return Estimate;
        }
    }

    // Once ingestion is complete, this search has already inspected every
    // remaining audio frame. If no close (or close/resume pair) was found,
    // waiting until playback finalization cannot add evidence and only strands
    // the uncommitted suffix behind a disproven text fence.
    if (Input.bInputStreamClosed
        && !Input.bPlaybackFinalized
        && (State.ActiveBoundaryMark == TEXT(',')
            || State.HoldDeadlinePlaybackSec > Input.ObservedAudioBufferEndSec + 0.001f))
    {
        Estimate.bContinuousSpeech = true;
        Estimate.Outcome = FName(TEXT("input_closed_no_decay_continuous"));
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

static int32 FindNextVowelPhoneIndex(const FOffgridAITextVisemePlan& Plan, int32 StartIndex)
{
    for (int32 PhoneIndex = FMath::Max(StartIndex, 0); PhoneIndex < Plan.ExpectedPhones.Num(); ++PhoneIndex)
    {
        if (Plan.ExpectedPhones[PhoneIndex].bIsVowel)
        {
            return PhoneIndex;
        }
    }
    return INDEX_NONE;
}

static int32 CountVowelPhonesThrough(const FOffgridAITextVisemePlan& Plan, int32 EndPhoneIndex)
{
    int32 Count = 0;
    for (int32 PhoneIndex = 0;
        PhoneIndex < Plan.ExpectedPhones.Num() && PhoneIndex <= EndPhoneIndex;
        ++PhoneIndex)
    {
        if (Plan.ExpectedPhones[PhoneIndex].bIsVowel) ++Count;
    }
    return Count;
}

static int32 CountVowelsAtOrBeforeActiveTime(
    const FOffgridAITextVisemePlan& Plan,
    const TArray<float>& PhoneCenterActiveSeconds,
    float ActiveSec)
{
    int32 Count = 0;
    for (int32 PhoneIndex = 0; PhoneIndex < Plan.ExpectedPhones.Num(); ++PhoneIndex)
    {
        if (PhoneCenterActiveSeconds.IsValidIndex(PhoneIndex)
            && Plan.ExpectedPhones[PhoneIndex].bIsVowel
            && PhoneCenterActiveSeconds[PhoneIndex] <= ActiveSec)
        {
            ++Count;
        }
    }
    return Count;
}

static float SmoothedPulseEnvelope(
    const TArray<FOffgridAIStreamingAudioFeatureFrame>& Frames,
    int32 CenterIndex,
    int32 HalfWidth)
{
    float WeightedSum = 0.0f;
    float WeightSum = 0.0f;
    for (int32 Offset = -HalfWidth; Offset <= HalfWidth; ++Offset)
    {
        const int32 FrameIndex = CenterIndex + Offset;
        if (!Frames.IsValidIndex(FrameIndex)) continue;
        const float Weight = static_cast<float>(HalfWidth + 1 - FMath::Abs(Offset));
        WeightedSum += Frames[FrameIndex].RMSNorm * Weight;
        WeightSum += Weight;
    }
    return WeightSum > 0.0f ? WeightedSum / WeightSum : 0.0f;
}

struct FCausalSyllableEnvelope
{
    float Prominence = 0.0f;
    float PeakSec = 0.0f;
    float StartSec = 0.0f;
    float CenterSec = 0.0f;
    float EndSec = 0.0f;
    float WidthSec = 0.0f;
};

static bool TryGetCausalSyllableEnvelope(
    const TArray<FOffgridAIStreamingAudioFeatureFrame>& Frames,
    int32 FrameIndex,
    FCausalSyllableEnvelope& OutEnvelope)
{
    if (!Frames.IsValidIndex(FrameIndex - 14) || !Frames.IsValidIndex(FrameIndex + 14)) return false;
    const float Center = SmoothedPulseEnvelope(Frames, FrameIndex, 6);
    if (Center < SmoothedPulseEnvelope(Frames, FrameIndex - 1, 6)
        || Center < SmoothedPulseEnvelope(Frames, FrameIndex + 1, 6))
    {
        return false;
    }
    float LeftFloor = Center;
    float RightFloor = Center;
    int32 LeftFloorIndex = FrameIndex;
    int32 RightFloorIndex = FrameIndex;
    for (int32 Index = FrameIndex - 14; Index <= FrameIndex - 6; ++Index)
    {
        const float Value = SmoothedPulseEnvelope(Frames, Index, 6);
        if (Value <= LeftFloor)
        {
            LeftFloor = Value;
            LeftFloorIndex = Index;
        }
    }
    for (int32 Index = FrameIndex + 6; Index <= FrameIndex + 14; ++Index)
    {
        const float Value = SmoothedPulseEnvelope(Frames, Index, 6);
        if (Value <= RightFloor)
        {
            RightFloor = Value;
            RightFloorIndex = Index;
        }
    }
    OutEnvelope.Prominence = FMath::Clamp(FMath::Min(Center - LeftFloor, Center - RightFloor) / 0.12f, 0.0f, 1.0f);
    const FOffgridAIStreamingAudioFeatureFrame& Frame = Frames[FrameIndex];
    const float StartSec = Frames[LeftFloorIndex].AudioBufferCenterSec;
    const float EndSec = Frames[RightFloorIndex].AudioBufferCenterSec;
    const float WidthSec = FMath::Max(EndSec - StartSec, 0.0f);
    const float CenterSec = 0.5f * (StartSec + EndSec);
    OutEnvelope.PeakSec = Frame.AudioBufferCenterSec;
    OutEnvelope.StartSec = StartSec;
    OutEnvelope.CenterSec = CenterSec;
    OutEnvelope.EndSec = EndSec;
    OutEnvelope.WidthSec = WidthSec;
    return OutEnvelope.Prominence >= 0.16f
        && Frame.SpeechEvidence >= 0.18f
        && Frame.Periodicity >= 0.20f
        && WidthSec >= 0.060f
        && WidthSec <= 0.320f;
}

static bool IsCausalSyllablePulse(
    const TArray<FOffgridAIStreamingAudioFeatureFrame>& Frames,
    int32 FrameIndex,
    float& OutProminence)
{
    FCausalSyllableEnvelope Envelope;
    if (!TryGetCausalSyllableEnvelope(Frames, FrameIndex, Envelope)) return false;
    OutProminence = Envelope.Prominence;
    return true;
}

static float SyllableVowelFamilyScore(
    const FString& PhoneBase,
    const FOffgridAIStreamingAudioFeatureFrame& Frame)
{
    const FOffgridAIArticulatoryProbabilityField Field =
        FOffgridAIAcousticEvidence::BuildArticulatoryProbabilityField(Frame);
    const bool bRound = PhoneBase == TEXT("UW") || PhoneBase == TEXT("UH")
        || PhoneBase == TEXT("OW") || PhoneBase == TEXT("OY") || PhoneBase == TEXT("AO");
    const bool bFront = PhoneBase == TEXT("IY") || PhoneBase == TEXT("IH")
        || PhoneBase == TEXT("EY") || PhoneBase == TEXT("EH") || PhoneBase == TEXT("AE");
    if (bRound)
    {
        return FMath::Clamp(
            Field.PhoneScores.VowelRound * 0.55f + Field.Vowel * 0.25f
                + Frame.Periodicity * 0.20f,
            0.0f,
            1.0f);
    }
    if (bFront)
    {
        return FMath::Clamp(
            Field.Vowel * 0.48f + Field.Voiced * 0.18f + Frame.Periodicity * 0.22f
                + (1.0f - Field.PhoneScores.VowelRound) * 0.12f,
            0.0f,
            1.0f);
    }
    return FMath::Clamp(
        Field.Vowel * 0.52f + Field.Voiced * 0.18f + Frame.Periodicity * 0.20f
            + Frame.LowBandNorm * 0.10f,
        0.0f,
        1.0f);
}

static int32 PhoneStress(const FString& Phone)
{
    if (Phone.Len() <= 0) return 0;
    const TCHAR Last = Phone[Phone.Len() - 1];
    return Last >= TCHAR('0') && Last <= TCHAR('2') ? Last - TCHAR('0') : 0;
}

static float CorpusSyllableIntervalScale(int32 PreviousStress, int32 CurrentStress)
{
    // Medians fitted on the first 80% of the checked-in corpus. They affect
    // assignment likelihood only; accepted anchors still derive timing from
    // streamed audio.
    static const float Scale[3][3] = {
        {0.917f, 1.123f, 0.944f},
        {1.035f, 1.067f, 1.111f},
        {1.041f, 1.257f, 1.000f},
    };
    return Scale[FMath::Clamp(PreviousStress, 0, 2)][FMath::Clamp(CurrentStress, 0, 2)];
}

static float StrongPhoneSupportAtFrame(
    const FString& PhoneBase,
    const FOffgridAIStreamingAudioFeatureFrame& Frame)
{
    const FOffgridAIArticulatoryProbabilityField Field =
        FOffgridAIAcousticEvidence::BuildArticulatoryProbabilityField(Frame);
    if (PhoneBase == TEXT("M") || PhoneBase == TEXT("B") || PhoneBase == TEXT("P"))
    {
        return FMath::Max(Field.PhoneScores.Bilabial, FMath::Max(Field.Closure, Field.Release));
    }
    if (PhoneBase == TEXT("F") || PhoneBase == TEXT("V"))
    {
        return FMath::Max(Field.PhoneScores.Labiodental, Field.Fricative);
    }
    if (PhoneBase == TEXT("W"))
    {
        return FMath::Max(Field.PhoneScores.Glide, Field.PhoneScores.VowelRound);
    }
    if (PhoneBase == TEXT("CH") || PhoneBase == TEXT("JH")
        || PhoneBase == TEXT("SH") || PhoneBase == TEXT("ZH"))
    {
        return FMath::Max(Field.PhoneScores.Sibilant, Field.PhoneScores.StopBurst);
    }
    if (PhoneBase == TEXT("UW") || PhoneBase == TEXT("UH") || PhoneBase == TEXT("OW")
        || PhoneBase == TEXT("OY") || PhoneBase == TEXT("AO"))
    {
        return FMath::Clamp(
            Field.PhoneScores.VowelRound * 0.60f + Field.Vowel * 0.20f
                + Frame.Periodicity * 0.20f,
            0.0f,
            1.0f);
    }
    return 0.0f;
}

static void ClearStrongPhoneRebase(FOffgridAIBoundaryPlaybackState& State)
{
    State.bStrongPhoneRebaseActive = false;
    State.StrongPhoneAnchorPhoneIndex = INDEX_NONE;
    State.StrongPhoneAnchorActiveSec = 0.0f;
    State.StrongPhoneAnchorAudioSec = 0.0f;
    State.StrongPhoneAnchorConfidence = 0.0f;
}

static void RefreshTranscriptAnchorFence(
    const FOffgridAITextVisemePlan& Plan,
    FOffgridAIBoundaryPlaybackState& State)
{
    State.TranscriptAnchorFenceWordIndex = INDEX_NONE;
    State.TranscriptAnchorFencePhoneIndex = INDEX_NONE;
    const int32 BoundaryCount = FMath::Min(
        Plan.WordBoundaryPunctuationAfter.Num(),
        Plan.WordPhoneEndIndices.Num());
    for (int32 WordIndex = 0; WordIndex < BoundaryCount; ++WordIndex)
    {
        if (WordIndex <= State.TranscriptAnchorLastResolvedBoundaryWordIndex) continue;
        const int32 FencePhoneIndex = Plan.WordPhoneEndIndices[WordIndex];
        if (FencePhoneIndex <= State.TranscriptAnchorCursorPhoneIndex) continue;
        const TCHAR Mark = Plan.WordBoundaryPunctuationAfter[WordIndex];
        const EOffgridAIBoundaryPauseClass PauseClass =
            Plan.WordBoundaryPauseClassAfter.IsValidIndex(WordIndex)
                ? Plan.WordBoundaryPauseClassAfter[WordIndex]
                : EOffgridAIBoundaryPauseClass::None;
        if (HoldSecondsForBoundaryClass(Mark, PauseClass) <= 0.0f) continue;
        State.TranscriptAnchorFenceWordIndex = WordIndex;
        State.TranscriptAnchorFencePhoneIndex = FencePhoneIndex;
        return;
    }
}

static void SynchronizeTranscriptAnchorCursor(
    const FOffgridAITextVisemePlan& Plan,
    int32 FirstMutablePhoneIndex,
    FOffgridAIBoundaryPlaybackState& State)
{
    if (State.LastResolvedBoundary.WordIndex != INDEX_NONE
        && State.LastResolvedBoundary.WordIndex
            > State.TranscriptAnchorLastResolvedBoundaryWordIndex)
    {
        State.TranscriptAnchorLastResolvedBoundaryWordIndex =
            State.LastResolvedBoundary.WordIndex;
        if (Plan.WordPhoneBeginIndices.IsValidIndex(State.LastResolvedBoundary.WordIndex + 1))
        {
            State.TranscriptAnchorCursorPhoneIndex = FMath::Max(
                State.TranscriptAnchorCursorPhoneIndex,
                Plan.WordPhoneBeginIndices[State.LastResolvedBoundary.WordIndex + 1]);
        }
        if (Plan.WordPhoneEndIndices.IsValidIndex(State.LastResolvedBoundary.WordIndex))
        {
            State.SyllableProgressCount = FMath::Max(
                State.SyllableProgressCount,
                CountVowelPhonesThrough(
                    Plan,
                    Plan.WordPhoneEndIndices[State.LastResolvedBoundary.WordIndex] - 1));
        }
    }
    RefreshTranscriptAnchorFence(Plan, State);
    int32 MutableCursor = FMath::Max(FirstMutablePhoneIndex, 0);
    if (State.TranscriptAnchorFencePhoneIndex != INDEX_NONE)
    {
        MutableCursor = FMath::Min(
            MutableCursor,
            FMath::Max(State.TranscriptAnchorFencePhoneIndex - 1, 0));
    }
    State.TranscriptAnchorCursorPhoneIndex = FMath::Max(
        State.TranscriptAnchorCursorPhoneIndex,
        MutableCursor);
}

static float SyllableStrongPhoneSupport(
    const FOffgridAITextVisemePlan& Plan,
    const TArray<float>& PhoneCenterActiveSeconds,
    const TArray<FOffgridAIStreamingAudioFeatureFrame>& Frames,
    int32 EnvelopeFrameIndex,
    int32 VowelPhoneIndex)
{
    if (!PhoneCenterActiveSeconds.IsValidIndex(VowelPhoneIndex)) return 0.0f;
    const float VowelActiveSec = PhoneCenterActiveSeconds[VowelPhoneIndex];
    float BestSupport = 0.0f;
    for (int32 PhoneIndex = FMath::Max(VowelPhoneIndex - 3, 0);
        PhoneIndex <= FMath::Min(VowelPhoneIndex + 3, Plan.ExpectedPhones.Num() - 1);
        ++PhoneIndex)
    {
        const FString& PhoneBase = Plan.ExpectedPhones[PhoneIndex].BasePhone;
        const float DirectSupport = StrongPhoneSupportAtFrame(PhoneBase, Frames[EnvelopeFrameIndex]);
        if (DirectSupport <= 0.0f) continue;
        const float OffsetSec = PhoneCenterActiveSeconds[PhoneIndex] - VowelActiveSec;
        const int32 ExpectedFrameIndex = EnvelopeFrameIndex + FMath::RoundToInt(OffsetSec / 0.010f);
        for (int32 FrameIndex = ExpectedFrameIndex - 4; FrameIndex <= ExpectedFrameIndex + 4; ++FrameIndex)
        {
            if (!Frames.IsValidIndex(FrameIndex)) continue;
            BestSupport = FMath::Max(BestSupport, StrongPhoneSupportAtFrame(PhoneBase, Frames[FrameIndex]));
        }
    }
    return BestSupport;
}

static void ResetSyllableRebaseForSection(
    FOffgridAIBoundaryPlaybackState& State,
    float SectionStartAudioSec,
    int32 LastDecidableFrameIndex)
{
    State.bSyllableRebaseActive = false;
    State.SyllableAnchorPhoneIndex = INDEX_NONE;
    State.SyllableRebaseEndPhoneIndex = INDEX_NONE;
    State.NextExpectedSyllablePhoneIndex = INDEX_NONE;
    State.LastSyllableScanFrameIndex = LastDecidableFrameIndex;
    State.SyllableAnchorActiveSec = 0.0f;
    State.SyllableAnchorAudioSec = 0.0f;
    State.SyllableRate = 1.0f;
    State.SyllablePlaybackRate = 1.0f;
    State.SyllableSectionStartAudioSec = FMath::Max(SectionStartAudioSec, 0.0f);
    State.SyllableAssignmentConfidence = 0.0f;
    State.SyllableAssignmentMargin = 0.0f;
    State.SyllableAssignmentSkipCount = 0;
    State.SyllableLastAssignedProminence = 0.0f;
    ClearStrongPhoneRebase(State);
}

static void ClearLocalSyllableRebase(FOffgridAIBoundaryPlaybackState& State)
{
    State.bSyllableRebaseActive = false;
    State.SyllableAnchorPhoneIndex = INDEX_NONE;
    State.SyllableRebaseEndPhoneIndex = INDEX_NONE;
    State.SyllableAnchorActiveSec = 0.0f;
    State.SyllableAnchorAudioSec = 0.0f;
    State.SyllableRate = 1.0f;
    State.SyllableAssignmentConfidence = 0.0f;
    State.SyllableAssignmentMargin = 0.0f;
    State.SyllableAssignmentSkipCount = 0;
    ClearStrongPhoneRebase(State);
}

static void UpdateStrongPhoneRebaseForSyllable(
    const FOffgridAITextVisemePlan& Plan,
    const TArray<float>& PhoneCenterActiveSeconds,
    const TArray<FOffgridAIStreamingAudioFeatureFrame>& Frames,
    int32 DecisionFrameIndex,
    int32 SyllablePhoneIndex,
    int32 NextSyllablePhoneIndex,
    float SyllableExpectedActiveSec,
    float SyllableObservedAudioSec,
    FOffgridAIBoundaryPlaybackState& State)
{
    ClearStrongPhoneRebase(State);
    const int32 SpanEnd = NextSyllablePhoneIndex == INDEX_NONE
        ? Plan.ExpectedPhones.Num()
        : NextSyllablePhoneIndex;
    const int32 SpanStart = FMath::Max(0, SyllablePhoneIndex - 3);
    float BestScore = 0.0f;
    float BestAudioSec = 0.0f;
    int32 BestPhoneIndex = INDEX_NONE;

    for (int32 PhoneIndex = SpanStart;
        PhoneIndex < SpanEnd && PhoneIndex <= SyllablePhoneIndex + 3;
        ++PhoneIndex)
    {
        if (!PhoneCenterActiveSeconds.IsValidIndex(PhoneIndex)) continue;
        const FString& PhoneBase = Plan.ExpectedPhones[PhoneIndex].BasePhone;
        if (StrongPhoneSupportAtFrame(PhoneBase, Frames[DecisionFrameIndex]) <= 0.0f) continue;
        ++State.StrongPhoneCandidateCount;
        const float ExpectedAudioSec = SyllableObservedAudioSec
            + (PhoneCenterActiveSeconds[PhoneIndex] - SyllableExpectedActiveSec);
        const float SearchHalfWidthSec =
            (PhoneBase == TEXT("M") || PhoneBase == TEXT("B") || PhoneBase == TEXT("P"))
                ? 0.090f : 0.075f;
        for (int32 FrameIndex = 0; FrameIndex <= DecisionFrameIndex; ++FrameIndex)
        {
            const float FrameSec = Frames[FrameIndex].AudioBufferCenterSec;
            const float DistanceSec = FMath::Abs(FrameSec - ExpectedAudioSec);
            if (DistanceSec > SearchHalfWidthSec) continue;
            const float Evidence = StrongPhoneSupportAtFrame(PhoneBase, Frames[FrameIndex]);
            const float Proximity = 1.0f - DistanceSec / SearchHalfWidthSec;
            const float Score = Evidence * 0.82f + Proximity * 0.18f;
            if (Score > BestScore)
            {
                BestScore = Score;
                BestAudioSec = FrameSec;
                BestPhoneIndex = PhoneIndex;
            }
        }
    }

    // The advisory experiment was useful but not precise enough to justify
    // broad corrections. Runtime accepts only its strongest subset.
    if (BestPhoneIndex == INDEX_NONE || BestScore < 0.90f) return;
    State.bStrongPhoneRebaseActive = true;
    State.StrongPhoneAnchorPhoneIndex = BestPhoneIndex;
    State.StrongPhoneAnchorActiveSec = PhoneCenterActiveSeconds[BestPhoneIndex];
    State.StrongPhoneAnchorAudioSec = BestAudioSec;
    State.StrongPhoneAnchorConfidence = BestScore;
    ++State.StrongPhoneAssignmentCount;
}

static void UpdateSyllableRebaseState(
    const FOffgridAILipsyncRuntimeUpdateInput& Input,
    const FOffgridAITextVisemePlan& Plan,
    const TArray<float>& PhoneCenterActiveSeconds,
    float PlaybackOffsetSec,
    int32 FirstMutablePhoneIndex,
    FOffgridAIBoundaryPlaybackState& State)
{
    if (State.bHoldActive || !Input.AudioFeatureFrames || Input.AudioFeatureFrames->Num() < 29) return;
    const TArray<FOffgridAIStreamingAudioFeatureFrame>& Frames = *Input.AudioFeatureFrames;
    const int32 LastDecidableFrame = Frames.Num() - 15;
    if (LastDecidableFrame <= State.LastSyllableScanFrameIndex) return;

    if (State.bSyllableRebaseActive
        && State.SyllableRebaseEndPhoneIndex != INDEX_NONE
        && FirstMutablePhoneIndex >= State.SyllableRebaseEndPhoneIndex)
    {
        ClearLocalSyllableRebase(State);
    }

    if (State.NextExpectedSyllablePhoneIndex == INDEX_NONE)
    {
        int32 StartPhoneIndex = 0;
        while (PhoneCenterActiveSeconds.IsValidIndex(StartPhoneIndex)
            && PhoneCenterActiveSeconds[StartPhoneIndex] + 0.080f < State.ActivePlayheadSec)
        {
            ++StartPhoneIndex;
        }
        State.NextExpectedSyllablePhoneIndex = FindNextVowelPhoneIndex(
            Plan,
            StartPhoneIndex);
    }

    struct FSyllableCandidate
    {
        int32 PhoneIndex = INDEX_NONE;
        int32 SkipCount = 0;
        float ExpectedActiveSec = 0.0f;
        float ExpectedAudioSec = 0.0f;
        float DistanceSec = 0.0f;
        float SequenceScore = 0.5f;
        float Score = 0.0f;
    };
    auto CommitAssignment = [&](const FSyllableCandidate& Selected,
        float SelectedMargin,
        float ObservedAudioSec,
        float Prominence,
        int32 DecisionFrameIndex,
        bool bTooLate)
    {
        State.SyllableAssignmentConfidence = Selected.Score;
        State.SyllableAssignmentMargin = SelectedMargin;
        State.SyllableAssignmentSkipCount = Selected.SkipCount;
        State.SyllableObservedAudioSec = ObservedAudioSec;
        State.SyllableLastAssignedProminence = Prominence;
        ++State.SyllableAssignmentCount;
        State.SyllableProgressCount = FMath::Max(
            State.SyllableProgressCount,
            CountVowelPhonesThrough(Plan, Selected.PhoneIndex));
        State.SyllableLastAssignedPhoneIndex = Selected.PhoneIndex;
        FOffgridAIRuntimeSyllableAssignmentDiagnosticRow AssignmentRow;
        AssignmentRow.PhoneIndex = Selected.PhoneIndex;
        AssignmentRow.ObservedAudioSec = ObservedAudioSec;
        AssignmentRow.Prominence = Prominence;
        AssignmentRow.Confidence = Selected.Score;
        AssignmentRow.Margin = SelectedMargin;
        AssignmentRow.SkipCount = Selected.SkipCount;
        State.PendingSyllableAssignments.Add(AssignmentRow);
        State.TranscriptAnchorCursorPhoneIndex = FMath::Max(
            State.TranscriptAnchorCursorPhoneIndex,
            Selected.PhoneIndex);
        if (SelectedMargin < 0.055f) ++State.SyllableAmbiguousAssignmentCount;

        State.NextExpectedSyllablePhoneIndex = FindNextVowelPhoneIndex(
            Plan,
            Selected.PhoneIndex + 1);
        UpdateStrongPhoneRebaseForSyllable(
            Plan,
            PhoneCenterActiveSeconds,
            Frames,
            DecisionFrameIndex,
            Selected.PhoneIndex,
            State.NextExpectedSyllablePhoneIndex,
            Selected.ExpectedActiveSec,
            ObservedAudioSec,
            State);
        if (bTooLate || Selected.PhoneIndex < FirstMutablePhoneIndex) return;

        float CorrectionStrength = FMath::Clamp((Selected.Score - 0.40f) / 0.32f, 0.0f, 1.0f);
        if (SelectedMargin < 0.0f) CorrectionStrength *= 0.35f;
        else if (SelectedMargin < 0.055f) CorrectionStrength *= 0.65f;
        const float AppliedAnchorAudioSec = Selected.ExpectedAudioSec
            + (ObservedAudioSec - Selected.ExpectedAudioSec) * CorrectionStrength;
        if (State.bSyllableRebaseActive)
        {
            const float PriorDelta = Selected.ExpectedActiveSec - State.SyllableAnchorActiveSec;
            const float AudioDelta = AppliedAnchorAudioSec - State.SyllableAnchorAudioSec;
            if (PriorDelta > 0.080f && AudioDelta > 0.030f)
            {
                const float Confidence = FMath::Clamp(
                    Selected.Score * (1.0f - Selected.DistanceSec / 0.350f), 0.0f, 1.0f);
                const float MeasuredRate = FMath::Clamp(AudioDelta / PriorDelta, 0.85f, 1.18f);
                State.SyllableRate = FMath::Clamp(
                    State.SyllableRate + (MeasuredRate - State.SyllableRate) * (0.18f + Confidence * 0.22f),
                    0.85f,
                    1.18f);
            }
        }
        State.bSyllableRebaseActive = true;
        State.SyllableAnchorPhoneIndex = Selected.PhoneIndex;
        State.SyllableAnchorActiveSec = Selected.ExpectedActiveSec;
        State.SyllableAnchorAudioSec = AppliedAnchorAudioSec;
        State.SyllableRebaseEndPhoneIndex = State.NextExpectedSyllablePhoneIndex;
    };

    for (int32 FrameIndex = FMath::Max(State.LastSyllableScanFrameIndex + 1, 14);
        FrameIndex <= LastDecidableFrame && State.NextExpectedSyllablePhoneIndex != INDEX_NONE;
        ++FrameIndex)
    {
        FCausalSyllableEnvelope Envelope;
        if (!TryGetCausalSyllableEnvelope(Frames, FrameIndex, Envelope)) continue;
        ++State.SyllablePulseCount;
        // Pulse detection itself is intentionally broad for diagnostics. Runtime
        // rebasing accepts a broader subset than strict landmarking because the
        // fit is local to a single syllable span and resets immediately after it.
        if (Envelope.Prominence < 0.20f)
        {
            ++State.SyllableRejectLowProminenceCount;
            continue;
        }
        const float EnvelopeCenterSec = Envelope.CenterSec;
        if (EnvelopeCenterSec + 0.001f < State.SyllableSectionStartAudioSec)
        {
            ++State.SyllableRejectBeforeSectionCount;
            continue;
        }
        // Historical evidence remains useful diagnostically, but it cannot
        // safely rebase live animation once its visible center is behind the
        // renderer's minimum lead window.
        const bool bPulseTooLate = !Input.bPlaybackFinalized
            && EnvelopeCenterSec < Input.CurrentPlaybackSec + 0.040f;
        if (bPulseTooLate)
        {
            ++State.SyllableLateAssignmentCount;
        }
        const int32 DuplicateGuardNextVowel = FindNextVowelPhoneIndex(
            Plan,
            State.NextExpectedSyllablePhoneIndex + 1);
        const bool bTranscriptAllowsRapidSyllable =
            PhoneCenterActiveSeconds.IsValidIndex(State.NextExpectedSyllablePhoneIndex)
            && PhoneCenterActiveSeconds.IsValidIndex(DuplicateGuardNextVowel)
            && PhoneCenterActiveSeconds[DuplicateGuardNextVowel]
                - PhoneCenterActiveSeconds[State.NextExpectedSyllablePhoneIndex] < 0.110f;
        if (State.SyllableAssignmentCount > 0
            && EnvelopeCenterSec - State.SyllableObservedAudioSec < 0.080f
            && Envelope.Prominence <= State.SyllableLastAssignedProminence * 1.50f
            && !bTranscriptAllowsRapidSyllable)
        {
            ++State.SyllableDuplicatePulseCount;
            continue;
        }

        TArray<FSyllableCandidate> Candidates;
        int32 CandidatePhoneIndex = State.NextExpectedSyllablePhoneIndex;
        for (int32 SkipCount = 0;
            SkipCount < 3 && PhoneCenterActiveSeconds.IsValidIndex(CandidatePhoneIndex);
            ++SkipCount)
        {
            if (State.TranscriptAnchorFencePhoneIndex != INDEX_NONE
                && CandidatePhoneIndex >= State.TranscriptAnchorFencePhoneIndex)
            {
                break;
            }
            const float ExpectedActiveSec = PhoneCenterActiveSeconds[CandidatePhoneIndex];
            float ExpectedAudioSec = PlaybackOffsetSec + ExpectedActiveSec;
            if (State.bResumeAnchorActive)
            {
                ExpectedAudioSec = State.ResumeAnchorFinalCenterSec
                    + (ExpectedActiveSec - State.ResumeAnchorActiveSec);
            }
            if (State.bSyllableRebaseActive)
            {
                ExpectedAudioSec = State.SyllableAnchorAudioSec
                    + State.SyllableRate * (ExpectedActiveSec - State.SyllableAnchorActiveSec);
            }
            const float DistanceSec = FMath::Abs(EnvelopeCenterSec - ExpectedAudioSec);
            const float TimingScore = FMath::Clamp(1.0f - DistanceSec / 0.350f, 0.0f, 1.0f);
            const float VowelScore = SyllableVowelFamilyScore(
                Plan.ExpectedPhones[CandidatePhoneIndex].BasePhone,
                Frames[FrameIndex]);
            float SequenceScore = 0.5f;
            if (State.SyllableAssignmentCount > 0
                && PhoneCenterActiveSeconds.IsValidIndex(State.SyllableLastAssignedPhoneIndex))
            {
                const float ExpectedIntervalSec = ExpectedActiveSec
                    - PhoneCenterActiveSeconds[State.SyllableLastAssignedPhoneIndex];
                const float ObservedIntervalSec = EnvelopeCenterSec - State.SyllableObservedAudioSec;
                if (ExpectedIntervalSec > 0.025f && ObservedIntervalSec > 0.0f)
                {
                    const int32 PreviousStress = PhoneStress(
                        Plan.ExpectedPhones[State.SyllableLastAssignedPhoneIndex].Phone);
                    const int32 CurrentStress = PhoneStress(
                        Plan.ExpectedPhones[CandidatePhoneIndex].Phone);
                    const float ExpectedScaledSec = ExpectedIntervalSec * State.SyllableRate
                        * CorpusSyllableIntervalScale(PreviousStress, CurrentStress);
                    const float IntervalToleranceSec = FMath::Clamp(
                        ExpectedScaledSec * 0.55f + 0.045f,
                        0.075f,
                        0.220f);
                    SequenceScore = FMath::Clamp(
                        1.0f - FMath::Abs(ObservedIntervalSec - ExpectedScaledSec)
                            / IntervalToleranceSec,
                        0.0f,
                        1.0f);
                }
            }
            const FOffgridAIArticulatoryProbabilityField EnvelopeField =
                FOffgridAIAcousticEvidence::BuildArticulatoryProbabilityField(Frames[FrameIndex]);
            const float NonNucleusEvidence = FMath::Max(
                EnvelopeField.Fricative,
                FMath::Max(EnvelopeField.Closure, EnvelopeField.Release));
            const float ContradictionScore = FMath::Clamp(
                NonNucleusEvidence - EnvelopeField.Vowel * 0.75f,
                0.0f,
                1.0f);
            const float StrongPhoneScore = SyllableStrongPhoneSupport(
                Plan,
                PhoneCenterActiveSeconds,
                Frames,
                FrameIndex,
                CandidatePhoneIndex);
            const FString& Phone = Plan.ExpectedPhones[CandidatePhoneIndex].Phone;
            const bool bPrimaryStress = Phone.Len() > 0 && Phone[Phone.Len() - 1] == TCHAR('1');
            const float StressScore = bPrimaryStress
                ? FMath::Clamp(Envelope.Prominence / 0.55f, 0.0f, 1.0f)
                : FMath::Clamp(1.0f - FMath::Abs(Envelope.Prominence - 0.42f), 0.0f, 1.0f);
            FSyllableCandidate Candidate;
            Candidate.PhoneIndex = CandidatePhoneIndex;
            Candidate.SkipCount = SkipCount;
            Candidate.ExpectedActiveSec = ExpectedActiveSec;
            Candidate.ExpectedAudioSec = ExpectedAudioSec;
            Candidate.DistanceSec = DistanceSec;
            Candidate.SequenceScore = SequenceScore;
            Candidate.Score = TimingScore * 0.40f + VowelScore * 0.23f
                + StrongPhoneScore * 0.15f + StressScore * 0.08f
                + SequenceScore * 0.14f - ContradictionScore * 0.10f
                - SkipCount * 0.08f;
            Candidates.Add(Candidate);
            CandidatePhoneIndex = FindNextVowelPhoneIndex(Plan, CandidatePhoneIndex + 1);
        }
        if (Candidates.Num() <= 0)
        {
            ++State.SyllableRejectNoCandidateCount;
            continue;
        }
        Candidates.Sort([](const FSyllableCandidate& A, const FSyllableCandidate& B)
        {
            return A.Score > B.Score;
        });
        const FSyllableCandidate* Primary = nullptr;
        const FSyllableCandidate* NextCandidate = nullptr;
        const FSyllableCandidate* ThirdCandidate = nullptr;
        for (const FSyllableCandidate& Candidate : Candidates)
        {
            if (Candidate.SkipCount == 0)
            {
                Primary = &Candidate;
            }
            else if (Candidate.SkipCount == 1)
            {
                NextCandidate = &Candidate;
            }
            else if (Candidate.SkipCount == 2)
            {
                ThirdCandidate = &Candidate;
            }
        }
        if (!Primary)
        {
            ++State.SyllableRejectNoCandidateCount;
            continue;
        }
        const float CompetitorScore = FMath::Max(
            NextCandidate ? NextCandidate->Score : 0.0f,
            ThirdCandidate ? ThirdCandidate->Score : 0.0f);
        const float Margin = Primary->Score - CompetitorScore;
        const FSyllableCandidate* Selected = Primary;
        for (const FSyllableCandidate& Candidate : Candidates)
        {
            if (Candidate.SkipCount <= 0) continue;
            const bool bStrongRecovery = Candidate.Score >= Primary->Score + 0.08f
                && Candidate.DistanceSec + 0.030f < Primary->DistanceSec;
            const bool bSequenceRecovery = Candidate.Score >= Primary->Score + 0.06f
                && Candidate.SequenceScore >= 0.65f;
            const bool bTimingRecovery = Candidate.SkipCount == 1
                && EnvelopeCenterSec - Primary->ExpectedAudioSec > 0.160f
                && Candidate.DistanceSec + 0.030f < Primary->DistanceSec;
            if (bStrongRecovery || bSequenceRecovery || bTimingRecovery)
            {
                Selected = &Candidate;
                break;
            }
        }
        const float SelectedMargin = Selected == Primary
            ? Margin
            : Selected->Score - Primary->Score;
        CommitAssignment(
            *Selected,
            SelectedMargin,
            EnvelopeCenterSec,
            Envelope.Prominence,
            FrameIndex,
            bPulseTooLate);
    }
    State.LastSyllableScanFrameIndex = LastDecidableFrame;
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
        // Audio owns the resumed center. Render spans may overlap across the
        // boundary; the event-center monotonicity guard below is the only
        // constraint when an earlier text schedule has drifted past the onset.
        const float AcousticBaseSec = InOutState.ObservedResumeOnsetPlaybackSec >= 0.0f
            ? InOutState.ObservedResumeOnsetPlaybackSec
            : AcousticAnchorSec;
        const float ObservedCenterSec = FMath::Max(AcousticBaseSec, 0.0f);
        // A resume can only be confirmed after its onset has streamed past. If
        // that decision arrives behind the live playback clock, scheduling the
        // suffix at the historical timestamp makes Unreal receive expired
        // events. Preserve the observed timestamp below for diagnostics, but
        // place the first newly committable center just ahead of playback.
        constexpr float MinLiveResumeLeadSec = 0.040f;
        const float DesiredCenterSec = FMath::Max(
            ObservedCenterSec,
            PlaybackSec + MinLiveResumeLeadSec);

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
        InOutState.ObservedResumeEnergyAnchorSec = AcousticAnchorSec;
        const int32 LastDecidableFrameIndex = Input.AudioFeatureFrames
            ? Input.AudioFeatureFrames->Num() - 15
            : INDEX_NONE;
        ResetSyllableRebaseForSection(InOutState, DesiredCenterSec, LastDecidableFrameIndex);
        InOutState.LastResolvedBoundary.DecaySec = InOutState.ConfirmedQuietStartPlaybackSec;
        InOutState.LastResolvedBoundary.ResumeOnsetSec = InOutState.ObservedResumeOnsetPlaybackSec;
        InOutState.LastResolvedBoundary.ResumeEnergyAnchorSec = AcousticAnchorSec;
    };

    bool bReleasedContinuousNoDecayThisTick = false;

    if (InOutState.bHoldActive)
    {
        UpdateProvisionalListPause(Input, PlaybackSec, InOutState);
        FCausalBoundaryFenceEstimate Fence;
        // Every live release must be justified by acoustic evidence. Elapsed
        // patience alone may not release either a hard or soft punctuation
        // fence; the estimator must confirm a close/resume pair or sustained
        // continuous speech.
        Fence = EvaluateCausalBoundaryFence(Input, InOutState, PlaybackSec);
        if (!Fence.bObservedResume && IsSentenceTerminalPunctuation(InOutState.ActiveBoundaryMark))
        {
            float RegionQuietSec = -1.0f;
            float RegionResumeSec = -1.0f;
            float RegionAnchorSec = -1.0f;
            if (FindNextSpeechRegionResumeAnchorSec(
                    Input.SpeechRegions,
                    InOutState.HoldStartSpeechRegionIndex,
                    MinDecayToResumeGapForBoundaryClass(InOutState.ActiveBoundaryMark, InOutState.ActivePauseClass),
                    RegionQuietSec,
                    RegionResumeSec,
                    RegionAnchorSec))
            {
                Fence.bObservedClose = true;
                Fence.bObservedResume = true;
                Fence.bContinuousSpeech = false;
                Fence.CloseSec = RegionQuietSec;
                Fence.ResumeOnsetSec = RegionResumeSec;
                Fence.ResumeAnchorSec = RegionAnchorSec;
                Fence.QuietRMSNorm = 0.0f;
                Fence.QuietEvidence = 0.0f;
                Fence.QuietRawRMS = 0.0f;
                Fence.Outcome = FName(TEXT("sentence_next_speech_region"));
            }
        }
        if (Fence.bObservedResume
            && Fence.bObservedClose
            && Fence.ResumeOnsetSec + 0.001f < Fence.CloseSec
                + MinDecayToResumeGapForBoundaryClass(InOutState.ActiveBoundaryMark, InOutState.ActivePauseClass))
        {
            Fence.bObservedResume = false;
            Fence.ResumeOnsetSec = -1.0f;
            Fence.ResumeAnchorSec = -1.0f;
            Fence.Outcome = FName(TEXT("resume_precedes_close_rejected"));
        }
        if (!Fence.bObservedResume && Input.bPlaybackFinalized)
        {
            float BufferedQuietSec = -1.0f;
            float BufferedResumeOnsetSec = -1.0f;
            float BufferedResumeAnchorSec = -1.0f;
            float BufferedQuietRMSNorm = 1.0f;
            float BufferedQuietEvidence = 1.0f;
            float BufferedQuietRawRMS = 1.0f;
            bool bFoundBufferedResume = FindBufferedBoundaryResumeAnchorSec(
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
            if (bFoundBufferedResume
                && BufferedResumeOnsetSec + 0.001f < BufferedQuietSec
                    + MinDecayToResumeGapForBoundaryClass(InOutState.ActiveBoundaryMark, InOutState.ActivePauseClass))
            {
                bFoundBufferedResume = false;
            }
            if (!bFoundBufferedResume && IsSentenceTerminalPunctuation(InOutState.ActiveBoundaryMark))
            {
                bFoundBufferedResume = FindNextSpeechRegionResumeAnchorSec(
                    Input.SpeechRegions,
                    InOutState.HoldStartSpeechRegionIndex,
                    MinDecayToResumeGapForBoundaryClass(InOutState.ActiveBoundaryMark, InOutState.ActivePauseClass),
                    BufferedQuietSec,
                    BufferedResumeOnsetSec,
                    BufferedResumeAnchorSec);
                if (bFoundBufferedResume)
                {
                    BufferedQuietRMSNorm = 0.0f;
                    BufferedQuietEvidence = 0.0f;
                    BufferedQuietRawRMS = 0.0f;
                }
            }
            // Every final-search source must obey the same causal ordering.
            // In particular, a terminal speech-region fallback must not pair
            // end-of-stream quiet with an earlier historical resume.
            if (bFoundBufferedResume
                && BufferedResumeOnsetSec + 0.001f < BufferedQuietSec
                    + MinDecayToResumeGapForBoundaryClass(
                        InOutState.ActiveBoundaryMark,
                        InOutState.ActivePauseClass))
            {
                bFoundBufferedResume = false;
            }
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
            // A provisional close can be superseded by the close/resume pair
            // that actually resolved the fence. Keep diagnostics and future
            // matching attached to that causal pair, not to stale quiet.
            InOutState.ConfirmedQuietStartPlaybackSec = Fence.CloseSec;
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
            // The speculative punctuation hold stopped commits while wall-clock
            // playback continued. If audio never confirms a close/resume, do not
            // expose the resulting backlog to the live scheduler: its lateness
            // guard would compress the suffix into 50 ms steps. Translate the
            // uncommitted schedule to resume just ahead of playback and preserve
            // every text-prior interval from that point onward.
            // Audio rejected the speculative punctuation pause. Translate the
            // uncommitted suffix just ahead of playback while preserving its
            // text-prior intervals. Releasing accumulated active-playhead debt
            // would force the backlog through the 50 ms monotonic floor and
            // create a visible burst after commas.
            if (InOutState.ActiveBoundaryMark == TEXT(','))
            {
                const float ResumeCenterSec = PlaybackSec + 0.040f;
                const float DesiredPausedSec =
                    ResumeCenterSec + InOutState.HoldResumeTargetLeadSec
                    - InOutState.PlaybackOriginSec
                    - InOutState.HoldResumeTargetActiveSec;
                InOutState.TotalPausedSec = FMath::Max(DesiredPausedSec, 0.0f);
                InOutState.ActivePlayheadSec = InOutState.HoldResumeTargetActiveSec;
            }
            else
            {
                InOutState.ActivePlayheadSec += FMath::Max(
                    PlaybackSec - InOutState.HoldStartPlaybackSec,
                    0.0f);
            }
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
        InOutState.ActivePlayheadSec += DeltaSec * InOutState.SyllablePlaybackRate;
    }

    InOutState.LastPlaybackSec = PlaybackSec;
}

struct FRuntimePhoneSchedule
{
    float BaseStart = 0.0f;
    float Center = 0.0f;
    float BaseEnd = 0.0f;
    bool bUsingResumeAnchorSchedule = false;
    bool bExactAcousticAnchorEvent = false;
    bool bUsedResumeAnchor = false;
    bool bUsedInitialSpeechAnchor = false;
    FName AcousticAnchorKind = NAME_None;
    float AcousticAnchorSec = -1.0f;
};

static FRuntimePhoneSchedule BuildRuntimePhoneSchedule(
    const FOffgridAIBoundaryPlaybackState& HoldState,
    int32 NextEventIndex,
    int32 SourcePhoneGlobalIndex,
    float PlaybackOffsetSec,
    float RequiredPhoneStartActiveSec,
    float RequiredActiveSec,
    float RequiredPhoneEndActiveSec,
    float PoseLeadSec)
{
    FRuntimePhoneSchedule Schedule;
    Schedule.BaseStart = PlaybackOffsetSec + RequiredPhoneStartActiveSec;
    Schedule.Center = PlaybackOffsetSec + RequiredActiveSec;
    Schedule.BaseEnd = PlaybackOffsetSec + RequiredPhoneEndActiveSec;

    const bool bScheduleUsesResumeAnchor = HoldState.bResumeAnchorActive
        && NextEventIndex >= HoldState.ResumeAnchorEventIndex;
    Schedule.bExactAcousticAnchorEvent = bScheduleUsesResumeAnchor
        && NextEventIndex == HoldState.ResumeAnchorEventIndex;

    if (bScheduleUsesResumeAnchor)
    {
        Schedule.bUsingResumeAnchorSchedule = true;
        const float LeadDeltaSec = PoseLeadSec - HoldState.ResumeAnchorLeadSec;
        Schedule.BaseStart = HoldState.ResumeAnchorFinalCenterSec
            + (RequiredPhoneStartActiveSec - HoldState.ResumeAnchorActiveSec)
            - LeadDeltaSec;
        Schedule.Center = HoldState.ResumeAnchorFinalCenterSec
            + (RequiredActiveSec - HoldState.ResumeAnchorActiveSec)
            - LeadDeltaSec;
        Schedule.BaseEnd = HoldState.ResumeAnchorFinalCenterSec
            + (RequiredPhoneEndActiveSec - HoldState.ResumeAnchorActiveSec)
            - LeadDeltaSec;

        if (Schedule.bExactAcousticAnchorEvent)
        {
            Schedule.bUsedResumeAnchor = true;
            Schedule.bUsedInitialSpeechAnchor = HoldState.bResumeAnchorFromInitialSpeech;
            Schedule.AcousticAnchorKind = Schedule.bUsedInitialSpeechAnchor
                ? FName(TEXT("initial_energy_anchor"))
                : FName(TEXT("punctuation_resume_energy_anchor"));
            Schedule.AcousticAnchorSec =
                (!Schedule.bUsedInitialSpeechAnchor && HoldState.LastResolvedBoundary.ResumeEnergyAnchorSec >= 0.0f)
                    ? HoldState.LastResolvedBoundary.ResumeEnergyAnchorSec
                    : HoldState.ResumeAnchorFinalCenterSec;
        }
    }

    const bool bWithinLocalSyllableRebaseSpan =
        HoldState.SyllableRebaseEndPhoneIndex == INDEX_NONE
        || SourcePhoneGlobalIndex < HoldState.SyllableRebaseEndPhoneIndex;
    const bool bUseSyllableRebase = HoldState.bSyllableRebaseActive
        && SourcePhoneGlobalIndex >= HoldState.SyllableAnchorPhoneIndex
        && bWithinLocalSyllableRebaseSpan
        && RequiredActiveSec >= HoldState.SyllableAnchorActiveSec
        && !Schedule.bExactAcousticAnchorEvent;
    if (bUseSyllableRebase)
    {
        Schedule.BaseStart = HoldState.SyllableAnchorAudioSec
            + HoldState.SyllableRate
                * (RequiredPhoneStartActiveSec - HoldState.SyllableAnchorActiveSec);
        Schedule.Center = HoldState.SyllableAnchorAudioSec
            + HoldState.SyllableRate
                * (RequiredActiveSec - HoldState.SyllableAnchorActiveSec);
        Schedule.BaseEnd = HoldState.SyllableAnchorAudioSec
            + HoldState.SyllableRate
                * (RequiredPhoneEndActiveSec - HoldState.SyllableAnchorActiveSec);
    }

    const bool bUseStrongPhoneRebase = HoldState.bStrongPhoneRebaseActive
        && bWithinLocalSyllableRebaseSpan
        && SourcePhoneGlobalIndex >= HoldState.StrongPhoneAnchorPhoneIndex
        && RequiredActiveSec >= HoldState.StrongPhoneAnchorActiveSec
        && !Schedule.bExactAcousticAnchorEvent;
    if (bUseStrongPhoneRebase)
    {
        Schedule.BaseStart = HoldState.StrongPhoneAnchorAudioSec
            + HoldState.SyllableRate
                * (RequiredPhoneStartActiveSec - HoldState.StrongPhoneAnchorActiveSec);
        Schedule.Center = HoldState.StrongPhoneAnchorAudioSec
            + HoldState.SyllableRate
                * (RequiredActiveSec - HoldState.StrongPhoneAnchorActiveSec);
        Schedule.BaseEnd = HoldState.StrongPhoneAnchorAudioSec
            + HoldState.SyllableRate
                * (RequiredPhoneEndActiveSec - HoldState.StrongPhoneAnchorActiveSec);
    }

    return Schedule;
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
    RuntimeSyllableAssignmentDiagnosticRows.Reset();
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
    for (FOffgridAIRuntimeSyllableAssignmentDiagnosticRow& AssignmentRow : PunctuationHoldState.PendingSyllableAssignments)
    {
        AssignmentRow.LineID = LineID;
        AssignmentRow.UpdateOrdinal = RuntimeCommitDiagnosticUpdateOrdinal;
        RuntimeSyllableAssignmentDiagnosticRows.Add(AssignmentRow);
    }
    PunctuationHoldState.PendingSyllableAssignments.Reset();
    // Boundary state is a timeline, not a final snapshot. Retaining one row per
    // update makes stalls and premature releases diagnosable; resetting here
    // previously erased every transition except finalization.
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
        BoundaryRow.bSyllableRebaseActive = PunctuationHoldState.bSyllableRebaseActive;
        BoundaryRow.SyllableAnchorPhoneIndex = PunctuationHoldState.SyllableAnchorPhoneIndex;
        BoundaryRow.NextExpectedSyllablePhoneIndex = PunctuationHoldState.NextExpectedSyllablePhoneIndex;
        BoundaryRow.SyllableAnchorActiveSec = PunctuationHoldState.SyllableAnchorActiveSec;
        BoundaryRow.SyllableAnchorAudioSec = PunctuationHoldState.SyllableAnchorAudioSec;
        BoundaryRow.SyllableRate = PunctuationHoldState.SyllableRate;
        BoundaryRow.SyllablePlaybackRate = PunctuationHoldState.SyllablePlaybackRate;
        BoundaryRow.SyllableSectionStartAudioSec = PunctuationHoldState.SyllableSectionStartAudioSec;
        BoundaryRow.SyllableAssignmentConfidence = PunctuationHoldState.SyllableAssignmentConfidence;
        BoundaryRow.SyllableAssignmentMargin = PunctuationHoldState.SyllableAssignmentMargin;
        BoundaryRow.SyllableAssignmentSkipCount = PunctuationHoldState.SyllableAssignmentSkipCount;
        BoundaryRow.SyllableObservedAudioSec = PunctuationHoldState.SyllableObservedAudioSec;
        BoundaryRow.SyllablePulseCount = PunctuationHoldState.SyllablePulseCount;
        BoundaryRow.SyllableAssignmentCount = PunctuationHoldState.SyllableAssignmentCount;
        BoundaryRow.SyllableProgressCount = PunctuationHoldState.SyllableProgressCount;
        BoundaryRow.SyllableLastAssignedPhoneIndex = PunctuationHoldState.SyllableLastAssignedPhoneIndex;
        BoundaryRow.SyllableRejectLowProminenceCount = PunctuationHoldState.SyllableRejectLowProminenceCount;
        BoundaryRow.SyllableRejectBeforeSectionCount = PunctuationHoldState.SyllableRejectBeforeSectionCount;
        BoundaryRow.SyllableLateAssignmentCount = PunctuationHoldState.SyllableLateAssignmentCount;
        BoundaryRow.SyllableRejectNoCandidateCount = PunctuationHoldState.SyllableRejectNoCandidateCount;
        BoundaryRow.SyllableAmbiguousAssignmentCount = PunctuationHoldState.SyllableAmbiguousAssignmentCount;
        BoundaryRow.SyllableDuplicatePulseCount = PunctuationHoldState.SyllableDuplicatePulseCount;
        BoundaryRow.bStrongPhoneRebaseActive = PunctuationHoldState.bStrongPhoneRebaseActive;
        BoundaryRow.StrongPhoneAnchorPhoneIndex = PunctuationHoldState.StrongPhoneAnchorPhoneIndex;
        BoundaryRow.StrongPhoneAnchorActiveSec = PunctuationHoldState.StrongPhoneAnchorActiveSec;
        BoundaryRow.StrongPhoneAnchorAudioSec = PunctuationHoldState.StrongPhoneAnchorAudioSec;
        BoundaryRow.StrongPhoneAnchorConfidence = PunctuationHoldState.StrongPhoneAnchorConfidence;
        BoundaryRow.StrongPhoneCandidateCount = PunctuationHoldState.StrongPhoneCandidateCount;
        BoundaryRow.StrongPhoneAssignmentCount = PunctuationHoldState.StrongPhoneAssignmentCount;
        BoundaryRow.TranscriptAnchorCursorPhoneIndex = PunctuationHoldState.TranscriptAnchorCursorPhoneIndex;
        BoundaryRow.TranscriptAnchorFenceWordIndex = PunctuationHoldState.TranscriptAnchorFenceWordIndex;
        BoundaryRow.TranscriptAnchorFencePhoneIndex = PunctuationHoldState.TranscriptAnchorFencePhoneIndex;
        BoundaryRow.TranscriptAnchorLastResolvedBoundaryWordIndex =
            PunctuationHoldState.TranscriptAnchorLastResolvedBoundaryWordIndex;
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

    // Runtime scheduling uses one monotonic transcript-anchor cursor:
    // 1. transcript owns viseme identity and order,
    // 2. punctuation close/resume owns strict section boundaries,
    // 3. accepted syllable/phone anchors may rebase only the uncommitted suffix
    //    and may not cross the cursor's unresolved punctuation fence,
    // 4. text/CMU priors supply duration and pacing between acoustic anchors,
    // 5. punctuation pauses future commits only: finish the pre-boundary prefix,
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

    const int32 FirstMutablePhoneIndex = Plan.Events.IsValidIndex(NextEventIndex)
        ? Plan.Events[NextEventIndex].SourcePhoneGlobalIndex
        : Plan.ExpectedPhones.Num();
    SynchronizeTranscriptAnchorCursor(Plan, FirstMutablePhoneIndex, InOutHoldState);
    UpdateSyllableRebaseState(
        Input,
        Plan,
        PhoneCenterActiveSeconds,
        PlaybackOffsetSec,
        FirstMutablePhoneIndex,
        InOutHoldState);
    if (InOutHoldState.SyllableAssignmentCount > 0 && !InOutHoldState.bHoldActive)
    {
        const int32 FenceTailPhoneIndex = InOutHoldState.TranscriptAnchorFencePhoneIndex - 1;
        const bool bNearPunctuationFence = PhoneCenterActiveSeconds.IsValidIndex(FenceTailPhoneIndex)
            && PhoneCenterActiveSeconds[FenceTailPhoneIndex]
                - InOutHoldState.ActivePlayheadSec <= 0.600f;
        if (bNearPunctuationFence)
        {
            InOutHoldState.SyllablePlaybackRate = 1.0f;
        }
        else
        {
            const int32 PlaybackSyllableCount = CountVowelsAtOrBeforeActiveTime(
                Plan,
                PhoneCenterActiveSeconds,
                InOutHoldState.ActivePlayheadSec);
            const int32 ProgressError = InOutHoldState.SyllableProgressCount
                - PlaybackSyllableCount;
            const float TargetRate = ProgressError >= 2 ? 1.04f
                : ProgressError == 1 ? 1.02f
                : ProgressError <= -2 ? 0.96f
                : ProgressError == -1 ? 0.98f
                : 1.0f;
            InOutHoldState.SyllablePlaybackRate = FMath::Clamp(
                InOutHoldState.SyllablePlaybackRate
                    + (TargetRate - InOutHoldState.SyllablePlaybackRate) * 0.20f,
                0.96f,
                1.04f);
        }
    }

    float CommitSafeActiveSec = FMath::Max(InOutHoldState.ActivePlayheadSec - CommitLagSec, 0.0f);

    int32 SchedulerIterationCount = 0;
    const int32 MaxSchedulerIterations = EventCount * 4 + 32;
    while (Plan.Events.IsValidIndex(NextEventIndex))
    {
        if (++SchedulerIterationCount > MaxSchedulerIterations)
        {
            break;
        }
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
                const int32 LastDecidableFrameIndex = Input.AudioFeatureFrames
                    ? Input.AudioFeatureFrames->Num() - 15
                    : INDEX_NONE;
                ResetSyllableRebaseForSection(
                    InOutHoldState,
                    InitialAnchorSec,
                    LastDecidableFrameIndex);
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
                // Hard punctuation may be reached after its acoustic pause has
                // already started. Search the bounded preroll for those strict
                // fences; keep commas local so they cannot attach to an older
                // list valley.
                const float BoundaryLookbackSec = Boundary == TEXT(',')
                    ? 0.120f
                    : (IsHardLikeBoundaryClass(Boundary, EffectivePauseClass)
                        ? FMath::Min(FMath::Max(Input.PrerollSec, 0.080f), 0.300f)
                        : 0.080f);
                const float PreviousBoundaryResumeFloorSec =
                    InOutHoldState.LastResolvedBoundary.ResumeOnsetSec >= 0.0f
                        ? InOutHoldState.LastResolvedBoundary.ResumeOnsetSec + 0.020f
                        : 0.0f;
                InOutHoldState.BoundarySearchStartPlaybackSec = FMath::Max(
                    FMath::Max(
                        Input.CurrentPlaybackSec - BoundaryLookbackSec,
                        PreviousBoundaryResumeFloorSec),
                    0.0f);
                InOutHoldState.HoldDeadlinePlaybackSec = Input.CurrentPlaybackSec + HoldSeconds;
                InOutHoldState.HoldStartSpeechRegionIndex = FindRegionIndexAtPlayback(EffectiveRegions, Input.CurrentPlaybackSec);
                InOutHoldState.bSawConfirmedOutOfSpeechAfterBoundary = false;
                InOutHoldState.ConfirmedQuietStartPlaybackSec = -1.0f;
                InOutHoldState.QuietRMSNormAtDecay = 1.0f;
                InOutHoldState.QuietEvidenceAtDecay = 1.0f;
                InOutHoldState.QuietRawRMSAtDecay = 1.0f;
                InOutHoldState.ActiveBoundaryMark = Boundary;
                InOutHoldState.bProvisionalListPause = false;
                InOutHoldState.ProvisionalListPauseStartSec = -1.0f;
                InOutHoldState.ProvisionalListPauseEndSec = -1.0f;
                InOutHoldState.ProvisionalListPauseConfidence = 0.0f;
                InOutHoldState.ProvisionalListPauseRMSNorm = 1.0f;
                InOutHoldState.ProvisionalListPauseEvidence = 1.0f;
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
                const int32 LastDecidableFrameIndex = Input.AudioFeatureFrames
                    ? Input.AudioFeatureFrames->Num() - 15
                    : INDEX_NONE;
                ResetSyllableRebaseForSection(
                    InOutHoldState,
                    Input.CurrentPlaybackSec,
                    LastDecidableFrameIndex);
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

        if (InOutHoldState.bResumeAnchorActive
            && NextEventIndex >= InOutHoldState.ResumeAnchorEventIndex
            && NextEventIndex == InOutHoldState.ResumeAnchorEventIndex)
        {
            // When the first post-boundary event is reached, refresh the active
            // origin from that event's current transcript timing. Later events
            // inherit the same acoustic anchor without drifting from the
            // resumed word/sentence onset.
            InOutHoldState.ResumeAnchorActiveSec = RequiredActiveSec;
            InOutHoldState.ResumeAnchorLeadSec = PoseLeadSec;
        }
        FRuntimePhoneSchedule Schedule = BuildRuntimePhoneSchedule(
            InOutHoldState,
            NextEventIndex,
            SourcePhoneGlobalIndex,
            PlaybackOffsetSec,
            RequiredPhoneStartActiveSec,
            RequiredActiveSec,
            RequiredPhoneEndActiveSec,
            PoseLeadSec);
        const int32 PreviousWordIndex = NextEventIndex > 0
            ? Plan.Events[NextEventIndex - 1].WordIndex
            : INDEX_NONE;
        if (LastCenter >= 0.0f && PreviousWordIndex == T.WordIndex
            && Schedule.Center > LastCenter + 0.220f)
        {
            FOffgridAIBoundaryPlaybackState PriorOnlyState = InOutHoldState;
            PriorOnlyState.bSyllableRebaseActive = false;
            PriorOnlyState.bStrongPhoneRebaseActive = false;
            const FRuntimePhoneSchedule PriorOnlySchedule = BuildRuntimePhoneSchedule(
                PriorOnlyState,
                NextEventIndex,
                SourcePhoneGlobalIndex,
                PlaybackOffsetSec,
                RequiredPhoneStartActiveSec,
                RequiredActiveSec,
                RequiredPhoneEndActiveSec,
                PoseLeadSec);
            if (PriorOnlySchedule.Center >= LastCenter + 0.001f
                && PriorOnlySchedule.Center < Schedule.Center)
            {
                Schedule = PriorOnlySchedule;
            }
        }
        float BaseStart = Schedule.BaseStart;
        float Center = Schedule.Center;
        float BaseEnd = Schedule.BaseEnd;
        const bool bExactAcousticAnchorEvent = Schedule.bExactAcousticAnchorEvent;
        const bool bUsingResumeAnchorSchedule = Schedule.bUsingResumeAnchorSchedule;
        const bool bUsedResumeAnchor = Schedule.bUsedResumeAnchor;
        const bool bUsedInitialSpeechAnchor = Schedule.bUsedInitialSpeechAnchor;
        const FName AcousticAnchorKind = Schedule.AcousticAnchorKind;
        const float AcousticAnchorSec = Schedule.AcousticAnchorSec;

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
        bool bLateSuffixRebase = false;
        if (!bPlaybackFinal)
        {
            const float CommitLeadSec = Center - Input.CurrentPlaybackSec;
            if (CommitLeadSec > MaxLiveLeadSec)
            {
                break;
            }

            // A stale event must not permanently block the entire suffix. Rebase
            // the mutable suffix just ahead of playback, preserving order and
            // allowing every later phone to inherit the same bounded correction.
            if (CommitLeadSec < -MaxLiveCommitBehindSec)
            {
                const float RecoveryCenterSec = Input.CurrentPlaybackSec + MinLiveLeadSec;
                const float RecoveryShiftSec = RecoveryCenterSec - Center;
                BaseStart += RecoveryShiftSec;
                Center = RecoveryCenterSec;
                BaseEnd += RecoveryShiftSec;
                InOutHoldState.bSyllableRebaseActive = true;
                InOutHoldState.SyllableAnchorPhoneIndex = SourcePhoneGlobalIndex;
                InOutHoldState.SyllableAnchorActiveSec = RequiredActiveSec;
                InOutHoldState.SyllableAnchorAudioSec = RecoveryCenterSec + PoseLeadSec;
                InOutHoldState.SyllableRebaseEndPhoneIndex =
                    InOutHoldState.TranscriptAnchorFencePhoneIndex;
                ClearStrongPhoneRebase(InOutHoldState);
                bLateSuffixRebase = true;
            }
        }

        FName EffectiveCommitReason = TextPriorMonotonicCommitReason;
        if (bLateSuffixRebase)
        {
            EffectiveCommitReason = FName(TEXT("late_suffix_rebase_commit"));
        }
        else if (bExactAcousticAnchorEvent)
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

        // Preserve a perceptible neutral interval across confirmed punctuation
        // pauses. The ordinary half-span anticipation is useful inside speech,
        // but on the first resumed event it can consume most of a short lull.
        // Keep a small coarticulation lead without beginning the pose well before
        // the detected resume onset.
        if (E.bUsedResumeAnchor && E.ObservedResumeOnsetSeconds >= 0.0f)
        {
            constexpr float ResumeRenderLeadSec = 0.020f;
            E.RenderStartSeconds = FMath::Max(
                E.RenderStartSeconds,
                E.ObservedResumeOnsetSeconds - ResumeRenderLeadSec);
        }

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
