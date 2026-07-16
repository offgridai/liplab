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

TArray<FOffgridAIStreamingSyllablePositionEstimate>
FOffgridAIStreamingSyllablePositionEstimator::Estimate(
    const FOffgridAITextVisemePlan& Plan,
    const TArray<FOffgridAIAudioLandmarkObservation>& EvidenceCandidates)
{
    const TArray<FSyllableTarget> Targets = BuildTargets(Plan);
    TArray<FOffgridAIAudioLandmarkObservation> Pulses;
    TArray<FOffgridAIAudioLandmarkObservation> PhoneCandidates;
    for (const auto& Observation : EvidenceCandidates)
    {
        if (Observation.Type == EOffgridAIAudioLandmarkType::SyllabicPulse) Pulses.Add(Observation);
        else if (IsPhoneFamily(Observation.Type)) PhoneCandidates.Add(Observation);
    }
    if (Targets.Num() <= 0 || Pulses.Num() <= 0) return {};

    const int32 Rows = Targets.Num();
    const int32 Columns = Pulses.Num();
    const float NegativeInfinity = -1.0e9f;
    constexpr float TargetSkipPenalty = 0.25f;
    constexpr float PulseSkipPenalty = 0.05f;
    constexpr float TimingWeight = 0.50f;
    constexpr float PhoneEvidenceWeight = 1.50f;
    constexpr float UnexpectedPhoneWeight = 0.10f;
    constexpr float PulseScoreWeight = 0.60f;
    constexpr float DurationWeight = 0.80f;
    constexpr float RhythmWeight = 0.15f;
    constexpr float DurationFloorSec = 0.100f;
    constexpr int32 MaxAssignmentStep = 2;
    auto Cell = [Columns](int32 Row, int32 Column) { return static_cast<size_t>(Row * Columns + Column); };
    const float PriorSpan = FMath::Max(Targets.Last().PriorCenterSec - Targets[0].PriorCenterSec, 0.001f);
    const float AudioSpan = FMath::Max(Pulses.Last().CenterSec - Pulses[0].CenterSec, 0.001f);

    std::vector<float> LocalScores(static_cast<size_t>(Rows * Columns), 0.0f);
    for (int32 Row = 0; Row < Rows; ++Row)
    {
        for (int32 Column = 0; Column < Columns; ++Column)
        {
            const float PriorProgress = (Targets[Row].PriorCenterSec - Targets[0].PriorCenterSec) / PriorSpan;
            const float AudioProgress = (Pulses[Column].CenterSec - Pulses[0].CenterSec) / AudioSpan;
            const float Timing = FMath::Clamp(
                1.0f - FMath::Abs(PriorProgress - AudioProgress) / 0.35f,
                0.0f,
                1.0f);
            LocalScores[Cell(Row, Column)] = MatchScore(
                Targets[Row],
                Pulses[Column],
                PhoneCandidates,
                Timing,
                0.00f,
                TimingWeight,
                PhoneEvidenceWeight,
                UnexpectedPhoneWeight)
                + PulseScoreWeight * Pulses[Column].Score;
        }
    }

    struct FContextState
    {
        int32 Row = INDEX_NONE;
        int32 Column = INDEX_NONE;
        int32 PreviousRow = INDEX_NONE;
        int32 PreviousColumn = INDEX_NONE;
        int32 BackState = INDEX_NONE;
        float Score = NegativeInfinity;
    };
    std::vector<FContextState> States;
    std::vector<std::vector<int32>> StateIndicesByCell(static_cast<size_t>(Rows * Columns));
    for (int32 Row = 0; Row < Rows; ++Row)
    {
        for (int32 Column = 0; Column < Columns; ++Column)
        {
            const int32 StateIndex = static_cast<int32>(States.size());
            States.push_back({
                Row,
                Column,
                INDEX_NONE,
                INDEX_NONE,
                INDEX_NONE,
                LocalScores[Cell(Row, Column)]
                    - TargetSkipPenalty * static_cast<float>(Row)
                    - PulseSkipPenalty * static_cast<float>(Column) });
            StateIndicesByCell[Cell(Row, Column)].push_back(StateIndex);
        }
    }

    for (int32 Row = 0; Row < Rows; ++Row)
    {
        for (int32 Column = 0; Column < Columns; ++Column)
        {
            const std::vector<int32> CurrentStateIndices = StateIndicesByCell[Cell(Row, Column)];
            for (const int32 CurrentStateIndex : CurrentStateIndices)
            {
                const FContextState Current = States[static_cast<size_t>(CurrentStateIndex)];
                for (int32 NextRow = Row + 1;
                    NextRow <= FMath::Min(Row + MaxAssignmentStep, Rows - 1);
                    ++NextRow)
                {
                    for (int32 NextColumn = Column + 1;
                        NextColumn <= FMath::Min(Column + MaxAssignmentStep, Columns - 1);
                        ++NextColumn)
                    {
                        const float ExpectedGap = Targets[NextRow].PriorCenterSec
                            - Targets[Row].PriorCenterSec;
                        const float ObservedGap = Pulses[NextColumn].CenterSec
                            - Pulses[Column].CenterSec;
                        const float DurationError = FMath::Min(
                            FMath::Abs(ObservedGap - ExpectedGap)
                                / FMath::Max(DurationFloorSec, ExpectedGap),
                            3.0f);
                        float TransitionCost = DurationWeight * DurationError;
                        if (Current.PreviousRow != INDEX_NONE
                            && Current.PreviousColumn != INDEX_NONE)
                        {
                            const float PreviousExpectedGap = Targets[Row].PriorCenterSec
                                - Targets[Current.PreviousRow].PriorCenterSec;
                            const float PreviousObservedGap = Pulses[Column].CenterSec
                                - Pulses[Current.PreviousColumn].CenterSec;
                            const float ExpectedRatio = static_cast<float>(std::log(
                                FMath::Max(ExpectedGap, 0.025f)
                                / FMath::Max(PreviousExpectedGap, 0.025f)));
                            const float ObservedRatio = static_cast<float>(std::log(
                                FMath::Max(ObservedGap, 0.025f)
                                / FMath::Max(PreviousObservedGap, 0.025f)));
                            TransitionCost += RhythmWeight
                                * FMath::Min(FMath::Abs(ExpectedRatio - ObservedRatio), 2.0f);
                        }
                        const float CandidateScore = Current.Score
                            + LocalScores[Cell(NextRow, NextColumn)]
                            - TargetSkipPenalty * static_cast<float>(NextRow - Row - 1)
                            - PulseSkipPenalty * static_cast<float>(NextColumn - Column - 1)
                            - TransitionCost;

                        int32 MatchingStateIndex = INDEX_NONE;
                        for (const int32 StateIndex : StateIndicesByCell[Cell(NextRow, NextColumn)])
                        {
                            const FContextState& State = States[static_cast<size_t>(StateIndex)];
                            if (State.PreviousRow == Row && State.PreviousColumn == Column)
                            {
                                MatchingStateIndex = StateIndex;
                                break;
                            }
                        }
                        if (MatchingStateIndex == INDEX_NONE)
                        {
                            MatchingStateIndex = static_cast<int32>(States.size());
                            States.push_back({
                                NextRow,
                                NextColumn,
                                Row,
                                Column,
                                CurrentStateIndex,
                                CandidateScore });
                            StateIndicesByCell[Cell(NextRow, NextColumn)].push_back(MatchingStateIndex);
                        }
                        else if (CandidateScore > States[static_cast<size_t>(MatchingStateIndex)].Score)
                        {
                            FContextState& State = States[static_cast<size_t>(MatchingStateIndex)];
                            State.Score = CandidateScore;
                            State.BackState = CurrentStateIndex;
                        }
                    }
                }
            }
        }
    }

    TArray<FOffgridAIStreamingSyllablePositionEstimate> Out;
    float BestFinal = NegativeInfinity;
    int32 BestStateIndex = INDEX_NONE;
    for (int32 StateIndex = 0; StateIndex < static_cast<int32>(States.size()); ++StateIndex)
    {
        const FContextState& State = States[static_cast<size_t>(StateIndex)];
        const float Final = State.Score
            - TargetSkipPenalty * static_cast<float>(Rows - State.Row - 1)
            - PulseSkipPenalty * static_cast<float>(Columns - State.Column - 1);
        if (Final <= BestFinal) continue;
        BestFinal = Final;
        BestStateIndex = StateIndex;
    }
    while (BestStateIndex != INDEX_NONE)
    {
        const FContextState& State = States[static_cast<size_t>(BestStateIndex)];
        const int32 Row = State.Row;
        const int32 Column = State.Column;
        const auto& Target = Targets[Row];
        const auto& Pulse = Pulses[Column];
        const float PriorProgress = (Target.PriorCenterSec - Targets[0].PriorCenterSec) / PriorSpan;
        const float AudioProgress = (Pulse.CenterSec - Pulses[0].CenterSec) / AudioSpan;
        const float Timing = FMath::Clamp(
            1.0f - FMath::Abs(PriorProgress - AudioProgress) / 0.35f,
            0.0f,
            1.0f);
        const float LocalScore = MatchScore(
            Target,
            Pulse,
            PhoneCandidates,
            Timing,
            0.00f,
            TimingWeight,
            PhoneEvidenceWeight,
            UnexpectedPhoneWeight)
            + PulseScoreWeight * Pulse.Score;
        FOffgridAIStreamingSyllablePositionEstimate Estimate;
        Estimate.SyllableIndex = Target.SyllableIndex;
        Estimate.NucleusPhoneIndex = Target.NucleusPhoneIndex;
        Estimate.WordIndex = Target.WordIndex;
        Estimate.SpeechRegionIndex = Target.SpeechRegionIndex;
        Estimate.AudioCenterSec = Pulse.CenterSec;
        Estimate.DecisionSec = Pulse.DecisionSec;
        Estimate.MatchScore = LocalScore;
        Estimate.Confidence = FMath::Clamp((LocalScore - 0.45f) / 1.20f, 0.0f, 1.0f);
        Out.Add(Estimate);
        BestStateIndex = State.BackState;
    }
    std::reverse(Out.begin(), Out.end());
    return Out;
}

TArray<FOffgridAIStreamingSyllablePositionEstimate>
FOffgridAIStreamingSyllablePositionEstimator::EstimateHistoricalAnchors(
    const FOffgridAITextVisemePlan& Plan,
    const TArray<FOffgridAIAudioLandmarkObservation>& EvidenceCandidates,
    const TArray<FOffgridAIStreamingSpeechRegion>& ObservedSpeechRegions,
    int32 RequiredStableUpdates,
    float MinMatchScore)
{
    const TArray<FSyllableTarget> Targets = BuildTargets(Plan);
    TArray<FOffgridAIAudioLandmarkObservation> Pulses;
    TArray<FOffgridAIAudioLandmarkObservation> PhoneCandidates;
    for (const auto& Observation : EvidenceCandidates)
    {
        if (Observation.Type == EOffgridAIAudioLandmarkType::SyllabicPulse) Pulses.Add(Observation);
        else if (IsPhoneFamily(Observation.Type)) PhoneCandidates.Add(Observation);
    }
    TArray<FOffgridAIStreamingSyllablePositionEstimate> Out;
    if (Targets.Num() <= 0 || Pulses.Num() <= 0) return Out;

    int32 ExpectedRegionCount = 0;
    for (const auto& Target : Targets)
        ExpectedRegionCount = FMath::Max(ExpectedRegionCount, Target.SpeechRegionIndex + 1);
    const bool bConstrainRegions = ExpectedRegionCount == ObservedSpeechRegions.Num();
    std::vector<int32> PulseRegions(static_cast<size_t>(Pulses.Num()), INDEX_NONE);
    if (bConstrainRegions)
    {
        for (int32 PulseIndex = 0; PulseIndex < Pulses.Num(); ++PulseIndex)
        {
            for (const auto& Region : ObservedSpeechRegions)
            {
                if (Pulses[PulseIndex].CenterSec >= Region.AudioBufferStartSec
                    && Pulses[PulseIndex].CenterSec <= Region.AudioBufferEndSec)
                {
                    PulseRegions[static_cast<size_t>(PulseIndex)] = Region.SpeechRegionIndex;
                    break;
                }
            }
        }
    }

    RequiredStableUpdates = FMath::Max(RequiredStableUpdates, 1);
    std::vector<int32> PreviousAssignments(static_cast<size_t>(Pulses.Num()), INDEX_NONE);
    std::vector<int32> StableUpdates(static_cast<size_t>(Pulses.Num()), 0);
    std::vector<bool> Accepted(static_cast<size_t>(Pulses.Num()), false);
    int32 LastAcceptedTarget = INDEX_NONE;

    for (int32 PrefixCount = 1; PrefixCount <= Pulses.Num(); ++PrefixCount)
    {
        const int32 Rows = Targets.Num() + 1;
        const int32 Columns = PrefixCount + 1;
        const float NegativeInfinity = -1.0e9f;
        std::vector<float> Scores(static_cast<size_t>(Rows * Columns), NegativeInfinity);
        std::vector<uint8> Back(static_cast<size_t>(Rows * Columns), 0);
        auto Cell = [Columns](int32 Row, int32 Column)
        {
            return static_cast<size_t>(Row * Columns + Column);
        };
        Scores[Cell(0, 0)] = 0.0f;
        const float CurrentDecisionSec = Pulses[PrefixCount - 1].DecisionSec;
        for (int32 Row = 0; Row < Rows; ++Row)
        {
            for (int32 Column = 0; Column < Columns; ++Column)
            {
                const float Current = Scores[Cell(Row, Column)];
                if (Current <= NegativeInfinity * 0.5f) continue;
                if (Row < Targets.Num() && Current - 0.35f > Scores[Cell(Row + 1, Column)])
                {
                    Scores[Cell(Row + 1, Column)] = Current - 0.35f;
                    Back[Cell(Row + 1, Column)] = 1;
                }
                if (Column < PrefixCount && Current - 0.25f > Scores[Cell(Row, Column + 1)])
                {
                    Scores[Cell(Row, Column + 1)] = Current - 0.25f;
                    Back[Cell(Row, Column + 1)] = 2;
                }
                if (Row < Targets.Num() && Column < PrefixCount)
                {
                    if (bConstrainRegions
                        && PulseRegions[static_cast<size_t>(Column)] != INDEX_NONE
                        && Targets[Row].SpeechRegionIndex
                            != PulseRegions[static_cast<size_t>(Column)])
                        continue;
                    const float PriorElapsed = Targets[Row].PriorCenterSec - Targets[0].PriorCenterSec;
                    const float AudioElapsed = Pulses[Column].CenterSec - Pulses[0].CenterSec;
                    const float Timing = FMath::Clamp(
                        1.0f - FMath::Abs(PriorElapsed - AudioElapsed) / 0.350f,
                        0.0f,
                        1.0f);
                    FOffgridAIAudioLandmarkObservation PulseForScore = Pulses[Column];
                    PulseForScore.DecisionSec = CurrentDecisionSec;
                    float LocalScore = MatchScore(
                        Targets[Row], PulseForScore, PhoneCandidates, Timing);
                    if (Row > 0 && Column > 0)
                    {
                        const float ExpectedGap = Targets[Row].PriorCenterSec
                            - Targets[Row - 1].PriorCenterSec;
                        const float ObservedGap = Pulses[Column].CenterSec
                            - Pulses[Column - 1].CenterSec;
                        const float DurationError = FMath::Min(
                            FMath::Abs(ObservedGap - ExpectedGap)
                                / FMath::Max(ExpectedGap, 0.100f),
                            3.0f);
                        LocalScore -= 0.10f * DurationError;
                        if (Row > 1 && Column > 1)
                        {
                            const float PreviousExpectedGap = Targets[Row - 1].PriorCenterSec
                                - Targets[Row - 2].PriorCenterSec;
                            const float PreviousObservedGap = Pulses[Column - 1].CenterSec
                                - Pulses[Column - 2].CenterSec;
                            const float ExpectedRatio = static_cast<float>(std::log(
                                FMath::Max(ExpectedGap, 0.025f)
                                / FMath::Max(PreviousExpectedGap, 0.025f)));
                            const float ObservedRatio = static_cast<float>(std::log(
                                FMath::Max(ObservedGap, 0.025f)
                                / FMath::Max(PreviousObservedGap, 0.025f)));
                            LocalScore -= 0.05f
                                * FMath::Min(FMath::Abs(ExpectedRatio - ObservedRatio), 2.0f);
                        }
                    }
                    if (Current + LocalScore > Scores[Cell(Row + 1, Column + 1)])
                    {
                        Scores[Cell(Row + 1, Column + 1)] = Current + LocalScore;
                        Back[Cell(Row + 1, Column + 1)] = 3;
                    }
                }
            }
        }
        int32 BestTerminalRow = 0;
        for (int32 CandidateRow = 1; CandidateRow < Rows; ++CandidateRow)
        {
            if (Scores[Cell(CandidateRow, PrefixCount)]
                > Scores[Cell(BestTerminalRow, PrefixCount)])
                BestTerminalRow = CandidateRow;
        }
        std::vector<int32> Assignments(static_cast<size_t>(PrefixCount), INDEX_NONE);
        std::vector<float> LocalScores(static_cast<size_t>(PrefixCount), 0.0f);
        int32 TraceRow = BestTerminalRow;
        int32 TraceColumn = PrefixCount;
        while (TraceRow > 0 || TraceColumn > 0)
        {
            const uint8 Step = Back[Cell(TraceRow, TraceColumn)];
            if (Step == 3)
            {
                Assignments[static_cast<size_t>(TraceColumn - 1)] = TraceRow - 1;
                LocalScores[static_cast<size_t>(TraceColumn - 1)] =
                    Scores[Cell(TraceRow, TraceColumn)]
                    - Scores[Cell(TraceRow - 1, TraceColumn - 1)];
                --TraceRow;
                --TraceColumn;
            }
            else if (Step == 1) --TraceRow;
            else if (Step == 2) --TraceColumn;
            else break;
        }

        for (int32 PulseIndex = 0; PulseIndex < PrefixCount; ++PulseIndex)
        {
            const int32 Assignment = Assignments[static_cast<size_t>(PulseIndex)];
            if (Assignment == INDEX_NONE)
            {
                PreviousAssignments[static_cast<size_t>(PulseIndex)] = INDEX_NONE;
                StableUpdates[static_cast<size_t>(PulseIndex)] = 0;
                continue;
            }
            if (PreviousAssignments[static_cast<size_t>(PulseIndex)] == Assignment)
                ++StableUpdates[static_cast<size_t>(PulseIndex)];
            else
            {
                PreviousAssignments[static_cast<size_t>(PulseIndex)] = Assignment;
                StableUpdates[static_cast<size_t>(PulseIndex)] = 1;
            }
            if (Accepted[static_cast<size_t>(PulseIndex)]
                || StableUpdates[static_cast<size_t>(PulseIndex)] < RequiredStableUpdates
                || LocalScores[static_cast<size_t>(PulseIndex)] < MinMatchScore
                || Assignment <= LastAcceptedTarget)
                continue;

            Accepted[static_cast<size_t>(PulseIndex)] = true;
            LastAcceptedTarget = Assignment;
            const auto& Target = Targets[Assignment];
            const float LocalScore = LocalScores[static_cast<size_t>(PulseIndex)];
            FOffgridAIStreamingSyllablePositionEstimate Estimate;
            Estimate.SyllableIndex = Assignment;
            Estimate.NucleusPhoneIndex = Target.NucleusPhoneIndex;
            Estimate.WordIndex = Target.WordIndex;
            Estimate.SpeechRegionIndex = Target.SpeechRegionIndex;
            Estimate.AudioCenterSec = Pulses[PulseIndex].CenterSec;
            Estimate.DecisionSec = CurrentDecisionSec;
            Estimate.MatchScore = LocalScore;
            Estimate.Confidence = FMath::Clamp(
                0.5f * (LocalScore - 0.45f) / 1.20f
                    + 0.5f * static_cast<float>(StableUpdates[static_cast<size_t>(PulseIndex)])
                        / static_cast<float>(RequiredStableUpdates + 1),
                0.0f,
                1.0f);
            Out.Add(Estimate);
        }
    }
    return Out;
}
