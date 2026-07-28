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

FOffgridAILipsyncPoseRuntimeState FOffgridAIVisemePerformer::BuildPoseStateFromSamples(
    const TArray<FOffgridAISubmittedVisemeSample>& Samples)
{
    FOffgridAILipsyncPoseRuntimeState State;
    for (const FOffgridAISubmittedVisemeSample& Sample : Samples)
    {
        if (Sample.PoseID.IsNone()) continue;
        if (!State.PoseWeights.Contains(Sample.PoseID))
            State.PoseIDs.Add(Sample.PoseID);
        float& Weight = State.PoseWeights.FindOrAdd(Sample.PoseID);
        Weight = FMath::Max(
            Weight,
            FMath::Clamp(Sample.SubmittedWeight, 0.0f, 1.0f));
    }
    return State;
}

void FOffgridAIVisemePerformer::BuildPoseWeightMapFromState(TMap<FName, float>& OutMap, const FOffgridAILipsyncPoseRuntimeState& State)
{
    OutMap.Reset();
    for (const FName& PoseID : State.PoseIDs)
    {
        if (PoseID.IsNone()) continue;
        OutMap.Add(
            PoseID,
            FMath::Clamp(State.PoseWeights.FindRef(PoseID), 0.0f, 1.0f));
    }
}

FOffgridAILipsyncPoseRuntimeState FOffgridAIVisemePerformer::StepDisplayedPose(
    const FOffgridAILipsyncPoseRuntimeState& Current,
    const FOffgridAILipsyncPoseRuntimeState& Target,
    float DeltaTimeSeconds,
    float RestClosedWeight,
    bool bLineEndingOrIdle)
{
    const float SafeDeltaTime = FMath::Max(DeltaTimeSeconds, 0.0f);
    auto StepPose = [SafeDeltaTime](
        float CurrentValue,
        float TargetValue,
        float AttackMs,
        float ReleaseMs) -> float
    {
        const float TauMs = TargetValue > CurrentValue ? AttackMs : ReleaseMs;
        const float TauSeconds = FMath::Max(TauMs * 0.001f, 0.001f);
        const float Alpha = FMath::Clamp(1.0f - FMath::Exp(-SafeDeltaTime / TauSeconds), 0.0f, 1.0f);
        return FMath::Clamp(FMath::Lerp(CurrentValue, TargetValue, Alpha), 0.0f, 1.0f);
    };

    const float EndMultiplier = bLineEndingOrIdle ? 0.45f : 1.0f;
    FOffgridAILipsyncPoseRuntimeState Out;
    auto AddPoseID = [&Out](const FName& PoseID)
    {
        if (!PoseID.IsNone() && !Out.PoseWeights.Contains(PoseID))
        {
            Out.PoseIDs.Add(PoseID);
            Out.PoseWeights.Add(PoseID, 0.0f);
        }
    };
    for (const FName& PoseID : Current.PoseIDs) AddPoseID(PoseID);
    for (const FName& PoseID : Target.PoseIDs) AddPoseID(PoseID);

    const FName ClosedPose(TEXT("22_MBP"));
    if (RestClosedWeight > 0.0f) AddPoseID(ClosedPose);
    for (const FName& PoseID : Out.PoseIDs)
    {
        float AttackMs = 32.0f;
        float ReleaseMs = 64.0f;
        if (PoseID == ClosedPose)
        {
            AttackMs = 18.0f;
            ReleaseMs = 46.0f;
        }
        else if (PoseID == FName(TEXT("12_Ww-Oo-"))
            || PoseID == FName(TEXT("20_FV"))
            || PoseID == FName(TEXT("21_FV-Ee-"))
            || PoseID == FName(TEXT("24_Tongue_Th")))
        {
            AttackMs = 24.0f;
            ReleaseMs = 54.0f;
        }
        else if (PoseID == FName(TEXT("09_Oh"))
            || PoseID == FName(TEXT("10_Or"))
            || PoseID == FName(TEXT("11_Oo"))
            || PoseID == FName(TEXT("17_Rr")))
        {
            AttackMs = 48.0f;
            ReleaseMs = 86.0f;
        }

        const float CurrentWeight = Current.PoseWeights.FindRef(PoseID);
        float TargetWeight = Target.PoseWeights.FindRef(PoseID);
        if (PoseID == ClosedPose)
            TargetWeight = FMath::Max(TargetWeight, RestClosedWeight);
        Out.PoseWeights.Add(
            PoseID,
            StepPose(
                CurrentWeight,
                TargetWeight,
                AttackMs,
                ReleaseMs * EndMultiplier));
    }
    return Out;
}
