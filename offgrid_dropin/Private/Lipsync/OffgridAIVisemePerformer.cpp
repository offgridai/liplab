#include "Lipsync/OffgridAIVisemePerformer.h"

namespace
{
static float SmoothStep01(float X)
{
    const float T = FMath::Clamp(X, 0.0f, 1.0f);
    return T * T * (3.0f - 2.0f * T);
}

static float MinimumJerk01(float X)
{
    const float T = FMath::Clamp(X, 0.0f, 1.0f);
    return T * T * T * (10.0f + T * (-15.0f + 6.0f * T));
}

static bool SameContinuousSpeechGroup(const FOffgridAICommittedVisemeEvent* A, const FOffgridAICommittedVisemeEvent* B)
{
    if (!A || !B) return false;
    if (A->SourcePhoneClass == FName(TEXT("acoustic_nucleus_beat"))
        || A->SourcePhoneClass == FName(TEXT("nucleus_gap"))
        || B->SourcePhoneClass == FName(TEXT("acoustic_nucleus_beat"))
        || B->SourcePhoneClass == FName(TEXT("nucleus_gap")))
        return false;
    if (B->bUsedResumeAnchor) return false;
    if (A->SpeechRegionIndex != INDEX_NONE && B->SpeechRegionIndex != INDEX_NONE && A->SpeechRegionIndex != B->SpeechRegionIndex) return false;

    // Runtime continuity follows the active detected speech region. Soft gaps
    // inside the same region may still keep a mouth state alive, but nothing
    // may bridge into an acoustically anchored punctuation resume.
    const float CenterGap = B->FinalRenderCenterSeconds - A->FinalRenderCenterSeconds;
    if (CenterGap > 0.180f)
    {
        return false;
    }
    return true;
}

static const FOffgridAICommittedVisemeEvent* FindPreviousRenderedEvent(
    const FOffgridAICommittedVisemeTrack& Track,
    int32 EventIndex)
{
    for (int32 I = EventIndex - 1; I >= 0; --I)
    {
        const FOffgridAICommittedVisemeEvent& Candidate = Track.Events[I];
        if (Candidate.bIsRenderable && !Candidate.bCanceledByWordHandoff)
            return &Candidate;
    }
    return nullptr;
}

static const FOffgridAICommittedVisemeEvent* FindNextRenderedEvent(
    const FOffgridAICommittedVisemeTrack& Track,
    int32 EventIndex)
{
    for (int32 I = EventIndex + 1; I < Track.Events.Num(); ++I)
    {
        const FOffgridAICommittedVisemeEvent& Candidate = Track.Events[I];
        if (Candidate.bIsRenderable && !Candidate.bCanceledByWordHandoff)
            return &Candidate;
    }
    return nullptr;
}

static float EventWeightAt(
    const FOffgridAICommittedVisemeEvent& E,
    const FOffgridAICommittedVisemeEvent* Prev,
    const FOffgridAICommittedVisemeEvent* Next,
    float PlaybackSeconds,
    float RegionStartSeconds,
    float RegionEndSeconds)
{
    // Strength is the planner's authoritative presentation magnitude. Rendering
    // uses one envelope for every pose and does not reinterpret strength by
    // phoneme or pose family.
    const bool bNucleusIndicator = E.SourcePhoneClass == FName(TEXT("acoustic_nucleus_beat"))
        || E.SourcePhoneClass == FName(TEXT("nucleus_gap"));
    const float Center = E.FinalRenderCenterSeconds;
    if (bNucleusIndicator)
    {
        constexpr float HalfWidth = 0.035f;
        const float Start = FMath::Max(Center - HalfWidth, RegionStartSeconds);
        const float End = FMath::Min(Center + HalfWidth, RegionEndSeconds);
        if (PlaybackSeconds < Start || PlaybackSeconds > End) return 0.0f;
        const float Shape = PlaybackSeconds <= Center
            ? SmoothStep01((PlaybackSeconds - Start) / FMath::Max(Center - Start, 0.001f))
            : 1.0f - SmoothStep01((PlaybackSeconds - Center) / FMath::Max(End - Center, 0.001f));
        return Shape * FMath::Clamp(E.Strength, 0.0f, 1.0f);
    }

    // Every committed pose gets a visible minimum envelope, including the
    // consonants that the old performer pruned. Prefer the neural token's
    // measured extent and add only a small presentation margin; precise
    // alignment should produce articulation, not a broad hold that bridges a
    // real pause.
    float AttackStart = FMath::Min(E.RenderStartSeconds - 0.015f, Center - 0.045f);
    float ReleaseEnd = FMath::Max(E.RenderEndSeconds + 0.015f, Center + 0.045f);

    // Adjacent gestures overlap around their midpoint. They do not occupy the
    // entire center-to-center interval: that former behavior made a modest
    // inference gap look like continuous speech and hid pause/resume errors.
    if (SameContinuousSpeechGroup(Prev, &E) && FMath::IsFinite(Prev->FinalRenderCenterSeconds))
    {
        const float PrevCenter = Prev->FinalRenderCenterSeconds;
        if (Center > PrevCenter)
            AttackStart = FMath::Min(
                AttackStart, 0.5f * (PrevCenter + Center) - 0.020f);
    }

    if (SameContinuousSpeechGroup(&E, Next) && Next && FMath::IsFinite(Next->FinalRenderCenterSeconds))
    {
        const float NextCenter = Next->FinalRenderCenterSeconds;
        if (NextCenter > Center)
            ReleaseEnd = FMath::Max(
                ReleaseEnd, 0.5f * (Center + NextCenter) + 0.020f);
    }

    // The speech detector owns animation gating. Neighbor blending may fill
    // soft gaps inside a region, but it must never leak into an inter-region
    // pause or anticipate a resume.
    AttackStart = FMath::Max(AttackStart, RegionStartSeconds);
    ReleaseEnd = FMath::Min(ReleaseEnd, RegionEndSeconds);

    if (PlaybackSeconds < AttackStart || PlaybackSeconds > ReleaseEnd)
    {
        return 0.0f;
    }

    // RenderStart/RenderEnd are the neural state's duration estimate. Fading
    // across each entire half-envelope discarded roughly one third of that
    // estimate below the visible threshold, making correctly scheduled words
    // look rushed. Restrict easing to short presentation edges and sustain the
    // state through the neural interior. Centers, ordering, and pause clamps
    // remain unchanged.
    constexpr float PresentationEdgeSec = 0.045f;
    const float AttackEnd = FMath::Min(Center, AttackStart + PresentationEdgeSec);
    const float ReleaseStart = FMath::Max(Center, ReleaseEnd - PresentationEdgeSec);
    float Shape = 1.0f;
    if (PlaybackSeconds < AttackEnd)
    {
        Shape = MinimumJerk01(
            (PlaybackSeconds - AttackStart)
            / FMath::Max(AttackEnd - AttackStart, 0.001f));
    }
    else if (PlaybackSeconds > ReleaseStart)
    {
        Shape = 1.0f - MinimumJerk01(
            (PlaybackSeconds - ReleaseStart)
            / FMath::Max(ReleaseEnd - ReleaseStart, 0.001f));
    }
    return Shape * FMath::Clamp(E.Strength, 0.0f, 1.0f);
}

static float JawOpenAt(
    const FOffgridAICommittedVisemeTrack& Track,
    float PlaybackSeconds)
{
    const FOffgridAICommittedVisemeTrack::FSpeechRegion* ActiveRegion = nullptr;
    for (const auto& Region : Track.SpeechRegions)
    {
        const float End = Region.bEnded
            ? Region.EndSeconds
            : TNumericLimits<float>::Max();
        if (PlaybackSeconds >= Region.StartSeconds - 0.001f
            && PlaybackSeconds <= End + 0.001f)
        {
            ActiveRegion = &Region;
            break;
        }
    }
    if (!ActiveRegion) return 0.0f;

    const FOffgridAICommittedVisemeEvent* Previous = nullptr;
    const FOffgridAICommittedVisemeEvent* Next = nullptr;
    for (const auto& Event : Track.Events)
    {
        if (Event.SpeechRegionIndex != ActiveRegion->SpeechRegionIndex
            || Event.JawOpenTarget < 0.0f
            || Event.bCanceledByWordHandoff
            || !FMath::IsFinite(Event.FinalRenderCenterSeconds))
            continue;
        if (Event.FinalRenderCenterSeconds <= PlaybackSeconds)
        {
            Previous = &Event;
        }
        else
        {
            Next = &Event;
            break;
        }
    }

    constexpr float JawAttackSec = 0.180f;
    // After the last nucleus the jaw begins closing through the syllable coda;
    // it does not hold the vowel posture until the detected region ends.
    constexpr float JawHoldSec = 0.000f;
    constexpr float JawReleaseSec = 0.140f;
    constexpr float ContinuousNucleusGapSec = 0.420f;

    if (Previous && Next)
    {
        const float PreviousCenter = Previous->FinalRenderCenterSeconds;
        const float NextCenter = Next->FinalRenderCenterSeconds;
        const float Gap = NextCenter - PreviousCenter;
        if (Gap <= ContinuousNucleusGapSec)
        {
            const float Phase = MinimumJerk01(
                (PlaybackSeconds - PreviousCenter)
                / FMath::Max(Gap, 0.001f));
            return FMath::Lerp(
                Previous->JawOpenTarget,
                Next->JawOpenTarget,
                Phase);
        }

        const float PreviousRelease = 1.0f - MinimumJerk01(
            (PlaybackSeconds - (PreviousCenter + JawHoldSec))
            / JawReleaseSec);
        const float NextAttack = MinimumJerk01(
            (PlaybackSeconds - (NextCenter - JawAttackSec))
            / JawAttackSec);
        return FMath::Max(
            Previous->JawOpenTarget * PreviousRelease,
            Next->JawOpenTarget * NextAttack);
    }

    if (Next)
    {
        const float AttackStart = FMath::Max(
            ActiveRegion->StartSeconds,
            Next->FinalRenderCenterSeconds - JawAttackSec);
        return Next->JawOpenTarget * MinimumJerk01(
            (PlaybackSeconds - AttackStart)
            / FMath::Max(
                Next->FinalRenderCenterSeconds - AttackStart,
                0.001f));
    }

    if (Previous)
    {
        const float ReleaseStart = Previous->FinalRenderCenterSeconds
            + JawHoldSec;
        float ReleaseEnd = ReleaseStart + JawReleaseSec;
        if (ActiveRegion->bEnded)
        {
            ReleaseEnd = FMath::Min(ReleaseEnd, ActiveRegion->EndSeconds);
        }
        return Previous->JawOpenTarget * (
            1.0f - MinimumJerk01(
                (PlaybackSeconds - ReleaseStart)
                / FMath::Max(ReleaseEnd - ReleaseStart, 0.001f)));
    }
    return 0.0f;
}
}

TArray<FOffgridAISubmittedVisemeSample> FOffgridAIVisemePerformer::Sample(const FOffgridAICommittedVisemeTrack& Track, float PlaybackSeconds, bool bGateBeforeSpeechStart)
{
    TArray<FOffgridAISubmittedVisemeSample> Out;
    if (bGateBeforeSpeechStart && Track.SpeechEndSeconds > Track.SpeechStartSeconds && PlaybackSeconds + 0.001f < Track.SpeechStartSeconds)
    {
        return Out;
    }
    for (int32 I = 0; I < Track.Events.Num(); ++I)
    {
        const FOffgridAICommittedVisemeEvent& E = Track.Events[I];
        if (!E.bIsRenderable || E.bCanceledByWordHandoff) continue;
        if (!FMath::IsFinite(E.FinalRenderCenterSeconds)) continue;
        const FOffgridAICommittedVisemeEvent* Prev = FindPreviousRenderedEvent(Track, I);
        const FOffgridAICommittedVisemeEvent* Next = FindNextRenderedEvent(Track, I);
        float RegionStartSeconds = Track.SpeechStartSeconds;
        float RegionEndSeconds = TNumericLimits<float>::Max();
        for (const auto& Region : Track.SpeechRegions)
        {
            if (Region.SpeechRegionIndex != E.SpeechRegionIndex) continue;
            RegionStartSeconds = Region.StartSeconds;
            if (Region.bEnded) RegionEndSeconds = Region.EndSeconds;
            break;
        }
        const float W = EventWeightAt(
            E, Prev, Next, PlaybackSeconds, RegionStartSeconds, RegionEndSeconds);
        if (W <= 0.012f) continue;
        FOffgridAISubmittedVisemeSample S;
        S.EventIndex = E.EventIndex;
        S.PoseID = E.PoseID;
        S.SourceWord = E.SourceWord;
        S.PlaybackSeconds = PlaybackSeconds;
        S.CommittedRenderStartSeconds = E.RenderStartSeconds;
        S.CommittedRenderCenterSeconds = E.FinalRenderCenterSeconds;
        S.CommittedRenderEndSeconds = E.RenderEndSeconds;
        S.SubmittedWeight = FMath::Clamp(W, 0.0f, 1.0f);
        S.SourceStrength = E.Strength;
        Out.Add(S);
    }
    // Jaw aperture shapes speech that the committed renderer is already
    // presenting; it must never manufacture activity across an otherwise
    // clean pause. This also keeps pause ownership in the single scheduler.
    float SpeechActivity = 0.0f;
    for (const FOffgridAISubmittedVisemeSample& Sample : Out)
    {
        SpeechActivity = FMath::Max(SpeechActivity, Sample.SubmittedWeight);
    }
    const float JawOpen = FMath::Min(
        JawOpenAt(Track, PlaybackSeconds),
        SpeechActivity);
    if (JawOpen > 0.012f)
    {
        FOffgridAISubmittedVisemeSample JawSample;
        JawSample.PoseID = TEXT("JawOpen");
        JawSample.SourceWord = TEXT("<syllabic-jaw-carrier>");
        JawSample.PlaybackSeconds = PlaybackSeconds;
        JawSample.SubmittedWeight = FMath::Clamp(JawOpen, 0.0f, 1.0f);
        JawSample.SourceStrength = JawSample.SubmittedWeight;
        Out.Add(JawSample);
    }
    // Neighboring gestures remain simultaneously available. The display
    // solver below supplies inertia, so coarticulation no longer depends on a
    // winner-takes-most pose switch at every event center.
    return Out;
}

TMap<FName, float> FOffgridAIVisemePerformer::CollapseByPoseID(const TArray<FOffgridAISubmittedVisemeSample>& Samples)
{
    // Multiple events for the same pose collapse by maximum weight. Cross-pose
    // clarity has already been handled once by Sample().
    TMap<FName, float> Out;
    for (const FOffgridAISubmittedVisemeSample& S : Samples)
    {
        if (S.PoseID.IsNone()) continue;
        float& V = Out.FindOrAdd(S.PoseID);
        V = FMath::Max(V, FMath::Clamp(S.SubmittedWeight, 0.0f, 1.0f));
    }
    return Out;
}

FOffgridAILipsyncPoseRuntimeState FOffgridAIVisemePerformer::BuildPoseStateFromPoseWeights(const TMap<FName, float>& PoseWeights)
{
    FOffgridAILipsyncPoseRuntimeState State;
    State.Closed = FMath::Clamp(FMath::Max(PoseWeights.FindRef(TEXT("22_MBP")), PoseWeights.FindRef(TEXT("MBP"))), 0.0f, 1.0f);
    if (const float* JawOpen = PoseWeights.Find(TEXT("JawOpen")))
    {
        State.Open = FMath::Clamp(*JawOpen, 0.0f, 1.0f);
    }
    else
    {
        // Compatibility for hosts submitting legacy pose maps.
        State.Open = FMath::Clamp(FMath::Max(FMath::Max(FMath::Max(PoseWeights.FindRef(TEXT("07_Aa")), PoseWeights.FindRef(TEXT("08_Ah"))), PoseWeights.FindRef(TEXT("18_Uh"))), PoseWeights.FindRef(TEXT("AAA"))), 0.0f, 1.0f);
    }
    State.Wide = FMath::Clamp(FMath::Max(FMath::Max(FMath::Max(PoseWeights.FindRef(TEXT("03_Ee")), PoseWeights.FindRef(TEXT("04_Ih"))), PoseWeights.FindRef(TEXT("05_Ay"))), PoseWeights.FindRef(TEXT("EEE"))), 0.0f, 1.0f);
    State.Round = FMath::Clamp(FMath::Max(FMath::Max(FMath::Max(PoseWeights.FindRef(TEXT("11_Oo")), PoseWeights.FindRef(TEXT("10_Or"))), PoseWeights.FindRef(TEXT("09_Oh"))), PoseWeights.FindRef(TEXT("OOO"))), 0.0f, 1.0f);
    State.Funnel = FMath::Clamp(FMath::Max(PoseWeights.FindRef(TEXT("12_Ww-Oo-")), PoseWeights.FindRef(TEXT("WUH"))), 0.0f, 1.0f);
    State.Teeth = FMath::Clamp(FMath::Max(PoseWeights.FindRef(TEXT("20_FV")), PoseWeights.FindRef(TEXT("FVS"))), 0.0f, 1.0f);
    return State;
}

void FOffgridAIVisemePerformer::BuildPoseWeightMapFromState(TMap<FName, float>& OutMap, const FOffgridAILipsyncPoseRuntimeState& State)
{
    OutMap.Reset();
    OutMap.Add(TEXT("MBP"), FMath::Clamp(State.Closed, 0.0f, 1.0f));
    OutMap.Add(TEXT("AAA"), FMath::Clamp(State.Open, 0.0f, 1.0f));
    OutMap.Add(TEXT("JawOpen"), FMath::Clamp(State.Open, 0.0f, 1.0f));
    OutMap.Add(TEXT("EEE"), FMath::Clamp(State.Wide, 0.0f, 1.0f));
    OutMap.Add(TEXT("OOO"), FMath::Clamp(State.Round, 0.0f, 1.0f));
    OutMap.Add(TEXT("WUH"), FMath::Clamp(State.Funnel, 0.0f, 1.0f));
    OutMap.Add(TEXT("FVS"), FMath::Clamp(State.Teeth, 0.0f, 1.0f));
    OutMap.Add(TEXT("LipCompression"), FMath::Clamp(State.Closed, 0.0f, 1.0f));
    OutMap.Add(TEXT("LipFunnel"), FMath::Clamp(State.Funnel, 0.0f, 1.0f));
    OutMap.Add(TEXT("LipStretch"), FMath::Clamp(State.Wide, 0.0f, 1.0f));
    OutMap.Add(TEXT("LowerLipDown"), FMath::Clamp(State.Teeth, 0.0f, 1.0f));
    OutMap.Add(TEXT("CornerPull"), 0.0f);
}

FOffgridAILipsyncPoseRuntimeState FOffgridAIVisemePerformer::StepDisplayedPose(
    const FOffgridAILipsyncPoseRuntimeState& Current,
    const FOffgridAILipsyncPoseRuntimeState& Target,
    float DeltaTimeSeconds,
    float RestClosedWeight,
    bool bLineEndingOrIdle)
{
    const float SafeDeltaTime = FMath::Max(DeltaTimeSeconds, 0.0f);
    auto StepPose = [SafeDeltaTime](float CurrentValue, float TargetValue, float AttackMs, float ReleaseMs) -> float
    {
        const float TauMs = TargetValue > CurrentValue ? AttackMs : ReleaseMs;
        const float TauSeconds = FMath::Max(TauMs * 0.001f, 0.001f);
        const float Alpha = FMath::Clamp(1.0f - FMath::Exp(-SafeDeltaTime / TauSeconds), 0.0f, 1.0f);
        return FMath::Clamp(FMath::Lerp(CurrentValue, TargetValue, Alpha), 0.0f, 1.0f);
    };

    const float EndMultiplier = bLineEndingOrIdle ? 0.45f : 1.0f;
    FOffgridAILipsyncPoseRuntimeState Out;
    Out.Closed = StepPose(Current.Closed, FMath::Max(Target.Closed, RestClosedWeight), 18.0f, FMath::Min(72.0f, 46.0f) * EndMultiplier);
    Out.Open = StepPose(Current.Open, Target.Open, 56.0f, FMath::Min(124.0f, 78.0f) * EndMultiplier);
    Out.Wide = StepPose(Current.Wide, Target.Wide, 56.0f, FMath::Min(124.0f, 78.0f) * EndMultiplier);
    Out.Round = StepPose(Current.Round, Target.Round, 48.0f, FMath::Min(140.0f, 86.0f) * EndMultiplier);
    Out.Funnel = StepPose(Current.Funnel, Target.Funnel, 24.0f, FMath::Min(84.0f, 54.0f) * EndMultiplier);
    Out.Teeth = StepPose(Current.Teeth, Target.Teeth, 24.0f, FMath::Min(84.0f, 54.0f) * EndMultiplier);
    return Out;
}
