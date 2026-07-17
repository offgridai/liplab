#include "Lipsync/OffgridAIVisemePerformer.h"

namespace
{
static float SmoothStep01(float X)
{
    const float T = FMath::Clamp(X, 0.0f, 1.0f);
    return T * T * (3.0f - 2.0f * T);
}

static bool SameContinuousSpeechGroup(const FOffgridAICommittedVisemeEvent* A, const FOffgridAICommittedVisemeEvent* B)
{
    if (!A || !B) return false;
    if (B->bUsedResumeAnchor) return false;
    if (A->SpeechRegionIndex != INDEX_NONE && B->SpeechRegionIndex != INDEX_NONE && A->SpeechRegionIndex != B->SpeechRegionIndex) return false;

    // Runtime continuity follows the active detected speech region. Soft gaps
    // inside the same region may still keep a mouth state alive, but nothing
    // may bridge into an acoustically anchored punctuation resume.
    const float CenterGap = B->FinalRenderCenterSeconds - A->FinalRenderCenterSeconds;
    if (CenterGap > 0.420f)
    {
        return false;
    }
    return true;
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
    constexpr float Attack = 0.055f;
    constexpr float Release = 0.070f;
    constexpr float HoldHalf = 0.030f;

    const float Center = E.FinalRenderCenterSeconds;
    const float PeakStart = Center - HoldHalf;
    const float PeakEnd = Center + HoldHalf;
    float AttackStart = PeakStart - Attack;
    float ReleaseEnd = PeakEnd + Release;

    // Treat committed visemes as states over a continuous speech region, not as
    // isolated impulses. The runtime commits a monotonic text-prior viseme prefix;
    // the performer should therefore keep a mouth state alive until the next
    // same-region state takes over. This fixes perceptual dead zones without
    // moving the committed event centers or adding a scheduler.
    if (SameContinuousSpeechGroup(Prev, &E) && FMath::IsFinite(Prev->FinalRenderCenterSeconds))
    {
        const float PrevCenter = Prev->FinalRenderCenterSeconds;
        if (Center > PrevCenter)
        {
            const float Boundary = (PrevCenter + Center) * 0.5f;
            AttackStart = FMath::Min(AttackStart, Boundary);
        }
    }

    if (SameContinuousSpeechGroup(&E, Next) && Next && FMath::IsFinite(Next->FinalRenderCenterSeconds))
    {
        const float NextCenter = Next->FinalRenderCenterSeconds;
        if (NextCenter > Center)
        {
            const float Boundary = (Center + NextCenter) * 0.5f;
            ReleaseEnd = FMath::Max(ReleaseEnd, Boundary);
        }
    }

    // The speech detector owns animation gating. Neighbor blending may fill
    // soft gaps inside a region, but it must never leak into an inter-region
    // pause or anticipate a resume.
    AttackStart = FMath::Max(AttackStart, RegionStartSeconds);
    ReleaseEnd = FMath::Min(ReleaseEnd, RegionEndSeconds);

    float Shape = 0.0f;
    if (PlaybackSeconds < AttackStart || PlaybackSeconds > ReleaseEnd)
    {
        return 0.0f;
    }
    if (PlaybackSeconds < PeakStart)
    {
        Shape = SmoothStep01((PlaybackSeconds - AttackStart) / FMath::Max(PeakStart - AttackStart, 0.001f));
    }
    else if (PlaybackSeconds <= PeakEnd)
    {
        Shape = 1.0f;
    }
    else
    {
        Shape = 1.0f - SmoothStep01((PlaybackSeconds - PeakEnd) / FMath::Max(ReleaseEnd - PeakEnd, 0.001f));
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
        if (!E.bIsRenderable) continue;
        if (!FMath::IsFinite(E.FinalRenderCenterSeconds)) continue;
        const FOffgridAICommittedVisemeEvent* Prev = I > 0 ? &Track.Events[I - 1] : nullptr;
        const FOffgridAICommittedVisemeEvent* Next = I + 1 < Track.Events.Num() ? &Track.Events[I + 1] : nullptr;
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
    // Keep one clearly readable primary articulation at a time. Neighbors are
    // retained for coarticulation, but cannot visually compete with the current
    // strongest pose. This is presentation-only: it does not move, remove, or
    // reorder committed events.
    int32 PrimaryIndex = INDEX_NONE;
    float PrimaryWeight = 0.0f;
    for (int32 I = 0; I < Out.Num(); ++I)
    {
        if (Out[I].SubmittedWeight > PrimaryWeight)
        {
            PrimaryWeight = Out[I].SubmittedWeight;
            PrimaryIndex = I;
        }
    }
    if (PrimaryIndex != INDEX_NONE)
    {
        constexpr float NeighborToPrimaryRatio = 0.20f;
        const FName PrimaryPose = Out[PrimaryIndex].PoseID;
        const float NeighborCap = PrimaryWeight * NeighborToPrimaryRatio;
        for (int32 I = 0; I < Out.Num(); ++I)
        {
            if (I == PrimaryIndex || Out[I].PoseID == PrimaryPose) continue;
            Out[I].SubmittedWeight = FMath::Min(Out[I].SubmittedWeight, NeighborCap);
        }
    }
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
    State.Open = FMath::Clamp(FMath::Max(FMath::Max(FMath::Max(PoseWeights.FindRef(TEXT("07_Aa")), PoseWeights.FindRef(TEXT("08_Ah"))), PoseWeights.FindRef(TEXT("18_Uh"))), PoseWeights.FindRef(TEXT("AAA"))), 0.0f, 1.0f);
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
