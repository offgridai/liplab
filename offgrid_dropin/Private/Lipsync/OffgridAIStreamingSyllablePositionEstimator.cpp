#include "Lipsync/OffgridAIStreamingSyllablePositionEstimator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace
{
struct FFamilyOffset
{
    EOffgridAIAudioLandmarkType Type = EOffgridAIAudioLandmarkType::SyllabicPulse;
    float OffsetSec = 0.0f;
};

struct FSyllableTarget
{
    int32 SyllableIndex = INDEX_NONE;
    int32 NucleusPhoneIndex = INDEX_NONE;
    int32 WordIndex = INDEX_NONE;
    int32 SpeechRegionIndex = INDEX_NONE;
    float PriorCenterSec = 0.0f;
    float PriorStartSec = 0.0f;
    float PriorEndSec = 0.0f;
    TArray<FFamilyOffset> FamilyOffsets;
};

static bool IsPhoneFamily(EOffgridAIAudioLandmarkType Type)
{
    return Type == EOffgridAIAudioLandmarkType::Bilabial
        || Type == EOffgridAIAudioLandmarkType::Labiodental
        || Type == EOffgridAIAudioLandmarkType::Glide
        || Type == EOffgridAIAudioLandmarkType::Sibilant
        || Type == EOffgridAIAudioLandmarkType::OpenVowel
        || Type == EOffgridAIAudioLandmarkType::FrontVowel
        || Type == EOffgridAIAudioLandmarkType::RoundedVowel;
}
static EOffgridAIAudioLandmarkType FamilyForPhone(const FString& BasePhone)
{
    if (BasePhone == TEXT("M") || BasePhone == TEXT("B") || BasePhone == TEXT("P"))
        return EOffgridAIAudioLandmarkType::Bilabial;
    if (BasePhone == TEXT("F") || BasePhone == TEXT("V"))
        return EOffgridAIAudioLandmarkType::Labiodental;
    if (BasePhone == TEXT("W") || BasePhone == TEXT("Y"))
        return EOffgridAIAudioLandmarkType::Glide;
    if (BasePhone == TEXT("S") || BasePhone == TEXT("Z") || BasePhone == TEXT("SH")
        || BasePhone == TEXT("ZH") || BasePhone == TEXT("CH") || BasePhone == TEXT("JH"))
        return EOffgridAIAudioLandmarkType::Sibilant;
    if (BasePhone == TEXT("AA") || BasePhone == TEXT("AE") || BasePhone == TEXT("AH")
        || BasePhone == TEXT("AW") || BasePhone == TEXT("AY"))
        return EOffgridAIAudioLandmarkType::OpenVowel;
    if (BasePhone == TEXT("EH") || BasePhone == TEXT("ER") || BasePhone == TEXT("EY")
        || BasePhone == TEXT("IH") || BasePhone == TEXT("IY"))
        return EOffgridAIAudioLandmarkType::FrontVowel;
    if (BasePhone == TEXT("AO") || BasePhone == TEXT("OW") || BasePhone == TEXT("OY")
        || BasePhone == TEXT("UH") || BasePhone == TEXT("UW"))
        return EOffgridAIAudioLandmarkType::RoundedVowel;
    return EOffgridAIAudioLandmarkType::SyllabicPulse;
}

static float FamilyReliability(EOffgridAIAudioLandmarkType Type)
{
    switch (Type)
    {
    case EOffgridAIAudioLandmarkType::Bilabial: return 0.58f;
    case EOffgridAIAudioLandmarkType::Labiodental: return 0.66f;
    case EOffgridAIAudioLandmarkType::Glide: return 0.77f;
    case EOffgridAIAudioLandmarkType::Sibilant: return 0.92f;
    case EOffgridAIAudioLandmarkType::OpenVowel: return 0.58f;
    case EOffgridAIAudioLandmarkType::FrontVowel: return 0.58f;
    case EOffgridAIAudioLandmarkType::RoundedVowel: return 0.66f;
    default: return 0.0f;
    }
}

static TArray<FSyllableTarget> BuildTargets(const FOffgridAITextVisemePlan& Plan)
{
    TArray<float> PhoneCenters;
    PhoneCenters.SetNum(Plan.ExpectedPhones.Num());
    float ElapsedSec = 0.0f;
    for (int32 Index = 0; Index < Plan.ExpectedPhones.Num(); ++Index)
    {
        const auto& Phone = Plan.ExpectedPhones[Index];
        PhoneCenters[Index] = ElapsedSec + Phone.WeightSeconds * 0.5f;
        ElapsedSec += Phone.WeightSeconds;
        if (Index + 1 < Plan.ExpectedPhones.Num()
            && Plan.ExpectedPhones[Index + 1].WordIndex != Phone.WordIndex
            && Plan.WordBoundaryPauseSecondsAfter.IsValidIndex(Phone.WordIndex))
        {
            ElapsedSec += Plan.WordBoundaryPauseSecondsAfter[Phone.WordIndex];
        }
    }

    TArray<FSyllableTarget> Targets;
    for (const FOffgridAIPlannedSyllable& Syllable : Plan.Syllables)
    {
        if (!Plan.ExpectedPhones.IsValidIndex(Syllable.NucleusPhoneIndex)) continue;
        const auto& Phone = Plan.ExpectedPhones[Syllable.NucleusPhoneIndex];
        FSyllableTarget Target;
        Target.SyllableIndex = Syllable.SyllableIndex;
        Target.NucleusPhoneIndex = Syllable.NucleusPhoneIndex;
        Target.WordIndex = Syllable.WordIndex;
        Target.SpeechRegionIndex = Syllable.SpeechRegionIndex;
        Target.PriorCenterSec = PhoneCenters[Syllable.NucleusPhoneIndex];
        Targets.Add(Target);
    }

    for (int32 PhoneIndex = 0; PhoneIndex < Plan.ExpectedPhones.Num(); ++PhoneIndex)
    {
        const auto& Phone = Plan.ExpectedPhones[PhoneIndex];
        const EOffgridAIAudioLandmarkType Family = FamilyForPhone(Phone.BasePhone);
        if (!IsPhoneFamily(Family)) continue;
        int32 BestTarget = INDEX_NONE;
        int32 BestDistance = TNumericLimits<int32>::Max();
        for (int32 TargetIndex = 0; TargetIndex < Targets.Num(); ++TargetIndex)
        {
            if (Targets[TargetIndex].WordIndex != Phone.WordIndex) continue;
            const int32 Distance = FMath::Abs(Targets[TargetIndex].NucleusPhoneIndex - Phone.PhoneIndex);
            if (Distance < BestDistance)
            {
                BestDistance = Distance;
                BestTarget = TargetIndex;
            }
        }
        if (BestTarget != INDEX_NONE)
        {
            Targets[BestTarget].FamilyOffsets.Add({
                Family,
                PhoneCenters[PhoneIndex] - Targets[BestTarget].PriorCenterSec });
        }
    }
    for (int32 TargetIndex = 0; TargetIndex < Targets.Num(); ++TargetIndex)
    {
        Targets[TargetIndex].PriorStartSec = TargetIndex > 0
            ? 0.5f * (Targets[TargetIndex - 1].PriorCenterSec + Targets[TargetIndex].PriorCenterSec)
            : FMath::Max(0.0f, Targets[TargetIndex].PriorCenterSec - 0.080f);
        Targets[TargetIndex].PriorEndSec = TargetIndex + 1 < Targets.Num()
            ? 0.5f * (Targets[TargetIndex].PriorCenterSec + Targets[TargetIndex + 1].PriorCenterSec)
            : Targets[TargetIndex].PriorCenterSec + 0.100f;
        if (Targets[TargetIndex].PriorEndSec <= Targets[TargetIndex].PriorStartSec)
            Targets[TargetIndex].PriorEndSec = Targets[TargetIndex].PriorStartSec + 0.080f;

    }
    return Targets;
}

static float MatchScore(
    const FSyllableTarget& Target,
    const FOffgridAIAudioLandmarkObservation& Pulse,
    const TArray<FOffgridAIAudioLandmarkObservation>& Candidates,
    float TimingScore,
    float BaseScore = 0.45f,
    float TimingWeight = 0.20f,
    float EvidenceWeight = 1.0f,
    float UnexpectedWeight = 0.20f)
{
    float ReliabilitySum = 0.0f;
    float EvidenceSum = 0.0f;
    for (const FFamilyOffset& Expected : Target.FamilyOffsets)
    {
        const float Reliability = FamilyReliability(Expected.Type);
        float Best = 0.0f;
        const float ExpectedCenter = Pulse.CenterSec + Expected.OffsetSec;
        for (const auto& Candidate : Candidates)
        {
            if (Candidate.Type != Expected.Type) continue;
            if (Candidate.DecisionSec > Pulse.DecisionSec + 0.001f) continue;
            const float Proximity = FMath::Clamp(
                1.0f - FMath::Abs(Candidate.CenterSec - ExpectedCenter) / 0.120f,
                0.0f,
                1.0f);
            Best = FMath::Max(Best, Candidate.Score * Proximity);
        }
        EvidenceSum += Reliability * Best;
        ReliabilitySum += Reliability;
    }
    const float Evidence = EvidenceSum / FMath::Max(ReliabilitySum, 1.0f);
    float Unexpected = 0.0f;
    for (const auto& Candidate : Candidates)
    {
        if (Candidate.DecisionSec > Pulse.DecisionSec + 0.001f
            || FMath::Abs(Candidate.CenterSec - Pulse.CenterSec) > 0.160f) continue;
        bool bExpected = false;
        for (const FFamilyOffset& Expected : Target.FamilyOffsets)
        {
            if (Expected.Type == Candidate.Type)
            {
                bExpected = true;
                break;
            }
        }
        if (!bExpected)
            Unexpected = FMath::Max(Unexpected, FamilyReliability(Candidate.Type) * Candidate.Score);
    }
    return BaseScore
        + TimingScore * TimingWeight
        + Evidence * EvidenceWeight
        - Unexpected * UnexpectedWeight;
}
}

TArray<FOffgridAIStreamingSyllableCandidateSet>
FOffgridAIStreamingSyllablePositionEstimator::EstimateCandidateSets(
    const FOffgridAITextVisemePlan& Plan,
    const TArray<FOffgridAIAudioLandmarkObservation>& EvidenceCandidates,
    int32 LookaheadSyllables,
    int32 BeamWidth,
    int32 MaxCandidates)
{
    const TArray<FSyllableTarget> Targets = BuildTargets(Plan);
    TArray<FOffgridAIAudioLandmarkObservation> Pulses;
    TArray<FOffgridAIAudioLandmarkObservation> PhoneCandidates;
    for (const auto& Observation : EvidenceCandidates)
    {
        if (Observation.Type == EOffgridAIAudioLandmarkType::SyllabicPulse)
            Pulses.Add(Observation);
        else if (IsPhoneFamily(Observation.Type))
            PhoneCandidates.Add(Observation);
    }

    TArray<FOffgridAIStreamingSyllableCandidateSet> Out;
    if (Targets.Num() <= 0 || Pulses.Num() <= 0) return Out;
    LookaheadSyllables = FMath::Max(LookaheadSyllables, 1);
    BeamWidth = FMath::Max(BeamWidth, 1);
    MaxCandidates = FMath::Max(MaxCandidates, 1);

    struct FBeamState
    {
        int32 NextTarget = 0;
        int32 LastTarget = INDEX_NONE;
        float LastPulseCenterSec = 0.0f;
        float Score = 0.0f;
    };
    struct FExpansion
    {
        FBeamState State;
        int32 AssignedTarget = INDEX_NONE;
        float AssignmentScore = -1.0e9f;
    };

    std::vector<FBeamState> Beam(1);
    for (const auto& Pulse : Pulses)
    {
        std::vector<FExpansion> Expansions;
        for (const FBeamState& State : Beam)
        {
            FExpansion Skip;
            Skip.State = State;
            Skip.State.Score -= 0.25f;
            Expansions.push_back(Skip);

            const int32 CandidateEnd = FMath::Min(
                State.NextTarget + LookaheadSyllables,
                Targets.Num());
            for (int32 TargetIndex = State.NextTarget;
                 TargetIndex < CandidateEnd;
                 ++TargetIndex)
            {
                float Timing = 0.50f;
                if (State.LastTarget != INDEX_NONE
                    && Targets.IsValidIndex(State.LastTarget))
                {
                    const float ExpectedGap = Targets[TargetIndex].PriorCenterSec
                        - Targets[State.LastTarget].PriorCenterSec;
                    const float AudioGap = Pulse.CenterSec - State.LastPulseCenterSec;
                    Timing = FMath::Clamp(
                        1.0f - FMath::Abs(ExpectedGap - AudioGap) / 0.250f,
                        0.0f,
                        1.0f);
                }
                const float LocalScore = MatchScore(
                    Targets[TargetIndex], Pulse, PhoneCandidates, Timing);
                const float SkippedTargetPenalty = 0.35f
                    * static_cast<float>(TargetIndex - State.NextTarget);

                FExpansion Assigned;
                Assigned.AssignedTarget = TargetIndex;
                Assigned.AssignmentScore = State.Score + LocalScore - SkippedTargetPenalty;
                Assigned.State.NextTarget = TargetIndex + 1;
                Assigned.State.LastTarget = TargetIndex;
                Assigned.State.LastPulseCenterSec = Pulse.CenterSec;
                Assigned.State.Score = Assigned.AssignmentScore;
                Expansions.push_back(Assigned);
            }
        }

        std::vector<std::pair<int32, float>> RankedCandidates;
        for (const FExpansion& Expansion : Expansions)
        {
            if (Expansion.AssignedTarget == INDEX_NONE) continue;
            auto Existing = std::find_if(
                RankedCandidates.begin(),
                RankedCandidates.end(),
                [&](const auto& Item) { return Item.first == Expansion.AssignedTarget; });
            if (Existing == RankedCandidates.end())
                RankedCandidates.emplace_back(Expansion.AssignedTarget, Expansion.AssignmentScore);
            else
                Existing->second = FMath::Max(Existing->second, Expansion.AssignmentScore);
        }
        std::sort(RankedCandidates.begin(), RankedCandidates.end(), [](const auto& A, const auto& B)
        {
            if (A.second != B.second) return A.second > B.second;
            return A.first < B.first;
        });

        FOffgridAIStreamingSyllableCandidateSet CandidateSet;
        CandidateSet.AudioCenterSec = Pulse.CenterSec;
        CandidateSet.DecisionSec = Pulse.DecisionSec;
        for (int32 Index = 0;
             Index < static_cast<int32>(RankedCandidates.size()) && Index < MaxCandidates;
             ++Index)
        {
            CandidateSet.SyllableIndices.Add(
                Targets[RankedCandidates[static_cast<size_t>(Index)].first].SyllableIndex);
            CandidateSet.Scores.Add(RankedCandidates[static_cast<size_t>(Index)].second);
        }
        Out.Add(CandidateSet);

        std::sort(Expansions.begin(), Expansions.end(), [](const FExpansion& A, const FExpansion& B)
        {
            if (A.State.Score != B.State.Score) return A.State.Score > B.State.Score;
            return A.State.NextTarget < B.State.NextTarget;
        });
        std::vector<FBeamState> NextBeam;
        for (const FExpansion& Expansion : Expansions)
        {
            const bool bDuplicate = std::any_of(
                NextBeam.begin(),
                NextBeam.end(),
                [&](const FBeamState& Kept)
                {
                    return Kept.NextTarget == Expansion.State.NextTarget;
                });
            if (bDuplicate) continue;
            NextBeam.push_back(Expansion.State);
            if (static_cast<int32>(NextBeam.size()) >= BeamWidth) break;
        }
        Beam = std::move(NextBeam);
    }
    return Out;
}
