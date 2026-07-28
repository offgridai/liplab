#include "Lipsync/OffgridAINeuralStreamingAligner.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace
{
constexpr float NegativeInfinity = -1.0e30f;

static FName NeuralPhoneClass(const FString& SourcePhone)
{
    const FString Phone = SourcePhone.ToUpper();
    if (Phone == TEXT("M") || Phone == TEXT("B") || Phone == TEXT("P"))
        return FName(TEXT("mbp"));
    if (Phone == TEXT("F") || Phone == TEXT("V")) return FName(TEXT("fv"));
    if (Phone == TEXT("W") || Phone == TEXT("Y")) return FName(TEXT("glide"));
    if (Phone == TEXT("S") || Phone == TEXT("Z") || Phone == TEXT("SH")
        || Phone == TEXT("ZH") || Phone == TEXT("CH") || Phone == TEXT("JH"))
        return FName(TEXT("sibilant"));
    if (Phone == TEXT("AA") || Phone == TEXT("AE") || Phone == TEXT("AH")
        || Phone == TEXT("AW") || Phone == TEXT("AY"))
        return FName(TEXT("open"));
    if (Phone == TEXT("EH") || Phone == TEXT("ER") || Phone == TEXT("EY")
        || Phone == TEXT("IH") || Phone == TEXT("IY"))
        return FName(TEXT("front"));
    if (Phone == TEXT("AO") || Phone == TEXT("OW") || Phone == TEXT("OY")
        || Phone == TEXT("UH") || Phone == TEXT("UW"))
        return FName(TEXT("round"));
    return FName(TEXT("other"));
}

static float StableSigmoid(float Value)
{
    if (Value >= 0.0f)
    {
        const float Exp = std::exp(-Value);
        return 1.0f / (1.0f + Exp);
    }
    const float Exp = std::exp(Value);
    return Exp / (1.0f + Exp);
}
}

bool FOffgridAINeuralStreamingAligner::Begin(
    const FOffgridAITextVisemePlan& Plan,
    const FString& CheckpointPath,
    FString& OutError)
{
    OutError.Reset();
    TextPlan = &Plan;
    Transcript = FOffgridAINeuralLipsyncBridge::BuildTranscriptTensor(Plan);
    if (Transcript.NumTokens() <= 0)
    {
        OutError = FString(TEXT("transcript produced no neural tokens"));
        return false;
    }
    if (CheckpointPath.IsEmpty())
    {
        OutError = FString(TEXT("neural checkpoint path is empty"));
        return false;
    }
    if ((!bCheckpointLoaded || LoadedCheckpointPath != CheckpointPath)
        && !Runtime.LoadCheckpoint(std::string(TCHAR_TO_UTF8(*CheckpointPath))))
    {
        OutError = FString(UTF8_TO_TCHAR(Runtime.LastError().c_str()));
        return false;
    }
    bCheckpointLoaded = true;
    LoadedCheckpointPath = CheckpointPath;
    if (!Runtime.ResetTokens(Transcript.Features.GetData(), Transcript.NumTokens()))
    {
        OutError = FString(UTF8_TO_TCHAR(Runtime.LastError().c_str()));
        return false;
    }

    DurationPriors.Init(0.040f, Transcript.NumTokens());
    for (int32 Token = 0; Token < Transcript.NumTokens(); ++Token)
    {
        const float* Row = Transcript.Features.GetData()
            + Token * offgridai::neural_streamer::kTokenDimensions;
        DurationPriors[Token] = FMath::Max(
            Row[offgridai::neural_streamer::kPhoneBuckets
                + offgridai::neural_streamer::kPoseBuckets],
            0.020f);
    }
    TokenFirstSec.Init(-1.0f, Transcript.NumTokens());
    TokenLastSec.Init(-1.0f, Transcript.NumTokens());
    TokenConfidence.Init(0.0f, Transcript.NumTokens());
    TokenFinished.Init(false, Transcript.NumTokens());
    PreviousCosts.assign(static_cast<size_t>(Transcript.NumTokens()), NegativeInfinity);
    CurrentCosts.assign(static_cast<size_t>(Transcript.NumTokens()), NegativeInfinity);
    PreviousDwells.assign(static_cast<size_t>(Transcript.NumTokens()), 0);
    CurrentDwells.assign(static_cast<size_t>(Transcript.NumTokens()), 0);
    Backtrace.clear();
    NextInputFrame = 0;
    NextCommitFrame = 0;
    LastAssignedToken = INDEX_NONE;
    bHasCommittedTrack = false;
    bFinalized = false;
    SpeechRegions.Reset();
    SpeechCandidateStartFrame = INDEX_NONE;
    QuietCandidateStartFrame = INDEX_NONE;
    bInSpeech = false;
    return true;
}

bool FOffgridAINeuralStreamingAligner::ProcessAvailableFrames(
    const TArray<FOffgridAIStreamingAudioFeatureFrame>& Frames,
    float CommitPlaybackSec,
    FOffgridAICommittedVisemeTrack& InOutTrack,
    FString& OutError)
{
    OutError.Reset();
    if (!TextPlan || bFinalized) return true;

    while (NextInputFrame < Frames.Num())
    {
        const int32 FrameCount = FMath::Min(
            offgridai::neural_streamer::kRuntimeChunkFrames,
            Frames.Num() - NextInputFrame);
        std::array<float,
            offgridai::neural_streamer::kRuntimeChunkFrames
                * offgridai::neural_streamer::kAudioFeatureCount> Audio{};
        for (int32 Offset = 0; Offset < FrameCount; ++Offset)
        {
            FOffgridAINeuralLipsyncBridge::BuildAudioFeatureVector(
                Frames[NextInputFrame + Offset],
                Audio.data() + Offset * offgridai::neural_streamer::kAudioFeatureCount);
        }

        std::vector<float> Scores(
            static_cast<size_t>(FrameCount) * Transcript.NumTokens(), 0.0f);
        std::array<float, offgridai::neural_streamer::kRuntimeChunkFrames> RegionLogits{};
        int ScoredTokenCount = 0;
        if (!Runtime.PushChunkAllTokens(
                Audio.data(),
                FrameCount,
                Scores.data(),
                ScoredTokenCount,
                RegionLogits.data()))
        {
            OutError = FString(UTF8_TO_TCHAR(Runtime.LastError().c_str()));
            return false;
        }

        for (int32 Offset = 0; Offset < FrameCount; ++Offset)
        {
            DecodeFrame(
                Scores.data() + static_cast<size_t>(Offset) * ScoredTokenCount,
                ScoredTokenCount,
                RegionLogits[Offset],
                Frames[NextInputFrame],
                CommitPlaybackSec,
                InOutTrack);
            ++NextInputFrame;
        }
    }
    RefreshTrackRegions(InOutTrack);
    return true;
}

void FOffgridAINeuralStreamingAligner::DecodeFrame(
    const float* Scores,
    int32 ScoredTokenCount,
    float RegionLogit,
    const FOffgridAIStreamingAudioFeatureFrame& Frame,
    float CommitPlaybackSec,
    FOffgridAICommittedVisemeTrack& InOutTrack)
{
    DecodeRegion(RegionLogit, Frame);
    const float SpeechProbability = FMath::Clamp(
        StableSigmoid(RegionLogit), 0.02f, 0.98f);
    std::fill(CurrentCosts.begin(), CurrentCosts.end(), NegativeInfinity);
    std::fill(CurrentDwells.begin(), CurrentDwells.end(), 0);

    float MaximumScore = Scores[0];
    for (int32 Offset = 1; Offset < ScoredTokenCount; ++Offset)
        MaximumScore = FMath::Max(MaximumScore, Scores[Offset]);
    float ScoreSum = 0.0f;
    std::vector<float> Probabilities(static_cast<size_t>(ScoredTokenCount), 0.0f);
    for (int32 Offset = 0; Offset < ScoredTokenCount; ++Offset)
    {
        Probabilities[static_cast<size_t>(Offset)] = std::exp(Scores[Offset] - MaximumScore);
        ScoreSum += Probabilities[static_cast<size_t>(Offset)];
    }
    ScoreSum = FMath::Max(ScoreSum, 1.0e-12f);

    FBacktraceFrame Record;
    Record.CenterSec = Frame.AudioBufferCenterSec;
    Record.PreviousToken.assign(static_cast<size_t>(ScoredTokenCount), INDEX_NONE);
    Record.Confidence.assign(static_cast<size_t>(ScoredTokenCount), 0.0f);

    for (int32 Offset = 0; Offset < ScoredTokenCount; ++Offset)
    {
        const int32 Token = Offset;
        const bool bSilenceToken =
            Transcript.SilenceTokens.IsValidIndex(Token)
            && Transcript.SilenceTokens[Token];
        const float OccupancyProbability = bSilenceToken
            ? 1.0f - SpeechProbability
            : SpeechProbability;
        // Both values are neural predictions. Fuse the independent occupancy
        // head into the token lattice so spoken transcript states cannot race
        // across a neural pause and leave the following speech region empty.
        constexpr float OccupancyLogWeight = 1.75f;
        const float Emission = Scores[Offset] - MaximumScore
            - std::log(ScoreSum)
            + OccupancyLogWeight * std::log(FMath::Clamp(
                OccupancyProbability, 0.02f, 0.98f));
        Record.Confidence[static_cast<size_t>(Offset)] =
            Probabilities[static_cast<size_t>(Offset)] / ScoreSum;
        if (Backtrace.empty())
        {
            if (Token == 0)
            {
                CurrentCosts[0] = Emission;
                CurrentDwells[0] = 1;
                Record.PreviousToken[static_cast<size_t>(Offset)] = 0;
            }
            continue;
        }

        const float AdvanceProbability = FMath::Clamp(
            0.010f / DurationPriors[Token], 0.02f, 0.80f);
        float Stay = PreviousCosts[Token] > NegativeInfinity * 0.5f
            ? PreviousCosts[Token] + std::log1p(-AdvanceProbability)
            : NegativeInfinity;
        const bool bCurrentSilence =
            Transcript.SilenceTokens.IsValidIndex(Token)
            && Transcript.SilenceTokens[Token];
        // The acoustic head may occasionally remain very confident in one
        // visible state for hundreds of milliseconds. Bound that dwell with
        // the learned/CMU duration prior so one pose cannot consume the rest
        // of a clause and force all following states into the tail window.
        // Silence is deliberately uncapped: neural occupancy still owns pause
        // duration and resume timing.
        const int32 MaximumDwellFrames = FMath::Clamp(
            static_cast<int32>(std::ceil(DurationPriors[Token] * 300.0f)),
            6,
            30);
        if (!bCurrentSilence
            && PreviousDwells[Token] >= MaximumDwellFrames)
        {
            Stay = NegativeInfinity;
        }
        float Advance = NegativeInfinity;
        const auto MinimumDwellFrames = [&](int32 Candidate) {
            if (Transcript.SentenceBoundaryTokens.IsValidIndex(Candidate)
                && Transcript.SentenceBoundaryTokens[Candidate])
                return 8;
            if (Transcript.SilenceTokens.IsValidIndex(Candidate)
                && Transcript.SilenceTokens[Candidate])
                return 1;
            return FMath::Clamp(
                static_cast<int32>(std::ceil(
                    DurationPriors[Candidate] * 50.0f)), 2, 8);
        };
        if (Token > 0
            && PreviousCosts[Token - 1] > NegativeInfinity * 0.5f)
        {
            const float PreviousAdvanceProbability = FMath::Clamp(
                0.010f / DurationPriors[Token - 1], 0.02f, 0.80f);
            const int32 MissingDwell = FMath::Max(
                MinimumDwellFrames(Token - 1) - PreviousDwells[Token - 1], 0);
            const bool bSentenceBoundary =
                Transcript.SentenceBoundaryTokens.IsValidIndex(Token - 1)
                && Transcript.SentenceBoundaryTokens[Token - 1];
            const float EarlyAdvancePenalty = MissingDwell
                * (bSentenceBoundary ? 0.75f : 0.50f);
            const bool bPreviousSilence =
                Transcript.SilenceTokens.IsValidIndex(Token - 1)
                && Transcript.SilenceTokens[Token - 1];
            // A visible pose needs at least two 10 ms acoustic frames before
            // the path may leave it. This is the smallest duration that can
            // survive the runtime visibility measurement and prevents a word
            // from collapsing into a stack of one-frame states.
            if (!bPreviousSilence && PreviousDwells[Token - 1] < 2)
            {
                Advance = NegativeInfinity;
            }
            else
            {
                Advance = PreviousCosts[Token - 1]
                    + std::log(PreviousAdvanceProbability)
                    - EarlyAdvancePenalty;
            }
        }
        if (Advance > Stay)
        {
            CurrentCosts[Token] = Advance + Emission;
            CurrentDwells[Token] = 1;
            Record.PreviousToken[static_cast<size_t>(Offset)] = Token - 1;
        }
        else if (Stay > NegativeInfinity * 0.5f)
        {
            CurrentCosts[Token] = Stay + Emission;
            CurrentDwells[Token] = PreviousDwells[Token] + 1;
            Record.PreviousToken[static_cast<size_t>(Offset)] = Token;
        }
    }
    PreviousCosts.swap(CurrentCosts);
    PreviousDwells.swap(CurrentDwells);
    Backtrace.push_back(Record);

    const int32 CurrentFrame = static_cast<int32>(Backtrace.size()) - 1;
    if (CurrentFrame >= FixedLagFrames && NextCommitFrame <= CurrentFrame - FixedLagFrames)
    {
        int32 State = BacktrackToFrame(BestCurrentToken(), NextCommitFrame);
        if (LastAssignedToken >= 0)
            State = FMath::Clamp(State, LastAssignedToken, LastAssignedToken + 1);
        bool bObservedNearFutureRegion = false;
        for (const FOffgridAIStreamingSpeechRegion& Region : SpeechRegions)
        {
            const float LeadSec = Region.AudioBufferStartSec
                - Backtrace[static_cast<size_t>(NextCommitFrame)].CenterSec;
            if (LeadSec > 0.020f && LeadSec <= 0.750f)
            {
                bObservedNearFutureRegion = true;
                break;
            }
        }
        const bool bObservedPauseEvidence = bObservedNearFutureRegion
            || !bInSpeech
            || QuietCandidateStartFrame != INDEX_NONE;
        if (bObservedPauseEvidence
            && Transcript.SilenceTokens.IsValidIndex(State)
            && !Transcript.SilenceTokens[State])
        {
            // Fixed-lag decoding can place the word after a comma just before
            // the acoustic pause and then let the following silence consume
            // the real gap. While neural region evidence says we are actually
            // inside that gap, keep the committed path on the transcript's
            // immediately preceding punctuation-pause token. The future region
            // is already observed inside the fixed-lag causal window; commas
            // without an acoustic split are unchanged. Never move backward
            // across an already committed state.
            for (int32 Candidate = State - 1;
                Candidate >= FMath::Max(LastAssignedToken, 0);
                --Candidate)
            {
                if (!Transcript.SilenceTokens.IsValidIndex(Candidate)
                    || !Transcript.SilenceTokens[Candidate])
                    continue;
                if (Transcript.PauseBoundaryTokens.IsValidIndex(Candidate)
                    && Transcript.PauseBoundaryTokens[Candidate])
                {
                    State = Candidate;
                }
                break;
            }
        }
        AssignCommittedFrame(
            State,
            Backtrace[static_cast<size_t>(NextCommitFrame)].CenterSec,
            ConfidenceAt(NextCommitFrame, State),
            CommitPlaybackSec,
            InOutTrack,
            FName(TEXT("neural_fixed_lag")));
        ++NextCommitFrame;
    }
}

void FOffgridAINeuralStreamingAligner::DecodeRegion(
    float RegionLogit,
    const FOffgridAIStreamingAudioFeatureFrame& Frame)
{
    constexpr float SpeechThreshold = 0.45f;
    constexpr int32 MinimumSpeechFrames = 2;
    constexpr int32 MinimumPauseFrames = 12;
    const bool bActive = StableSigmoid(RegionLogit) >= SpeechThreshold;
    const int32 FrameIndex = static_cast<int32>(Backtrace.size());

    if (!bInSpeech)
    {
        if (!bActive)
        {
            SpeechCandidateStartFrame = INDEX_NONE;
            return;
        }
        if (SpeechCandidateStartFrame == INDEX_NONE)
        {
            SpeechCandidateStartFrame = FrameIndex;
            SpeechCandidateStartSec = Frame.AudioBufferStartSec;
        }
        if (FrameIndex - SpeechCandidateStartFrame + 1 < MinimumSpeechFrames) return;

        FOffgridAIStreamingSpeechRegion Region;
        Region.SpeechRegionIndex = SpeechRegions.Num();
        Region.AudioBufferStartSec = FMath::Max(SpeechCandidateStartSec, 0.0f);
        Region.AudioBufferLastSpeechSec = Frame.AudioBufferEndSec;
        Region.AudioBufferEndSec = Frame.AudioBufferEndSec;
        Region.bStarted = true;
        Region.EndReason = FName(TEXT("neural_open"));
        SpeechRegions.Add(Region);
        bInSpeech = true;
        SpeechCandidateStartFrame = INDEX_NONE;
        QuietCandidateStartFrame = INDEX_NONE;
        return;
    }

    FOffgridAIStreamingSpeechRegion& Region = SpeechRegions.Last();
    if (bActive)
    {
        if (QuietCandidateStartFrame != INDEX_NONE) ++Region.ReopenCount;
        QuietCandidateStartFrame = INDEX_NONE;
        Region.AudioBufferLastSpeechSec = Frame.AudioBufferEndSec;
        Region.AudioBufferEndSec = Frame.AudioBufferEndSec;
        return;
    }
    if (QuietCandidateStartFrame == INDEX_NONE)
    {
        QuietCandidateStartFrame = FrameIndex;
        QuietCandidateStartSec = Frame.AudioBufferStartSec;
    }
    if (FrameIndex - QuietCandidateStartFrame + 1 < MinimumPauseFrames) return;

    Region.AudioBufferEndSec = FMath::Max(Region.AudioBufferStartSec, QuietCandidateStartSec);
    Region.ProvisionalEndSec = QuietCandidateStartSec;
    Region.EndDecisionSec = Frame.AudioBufferEndSec;
    Region.bEnded = true;
    Region.EndReason = FName(TEXT("neural_pause_120ms"));
    bInSpeech = false;
    QuietCandidateStartFrame = INDEX_NONE;
    SpeechCandidateStartFrame = INDEX_NONE;
}

int32 FOffgridAINeuralStreamingAligner::BestCurrentToken() const
{
    if (PreviousCosts.empty()) return INDEX_NONE;
    return static_cast<int32>(std::distance(
        PreviousCosts.begin(),
        std::max_element(PreviousCosts.begin(), PreviousCosts.end())));
}

int32 FOffgridAINeuralStreamingAligner::BacktrackToFrame(
    int32 Token,
    int32 TargetFrame) const
{
    if (Token < 0 || Backtrace.empty()) return FMath::Max(LastAssignedToken, 0);
    for (int32 Frame = static_cast<int32>(Backtrace.size()) - 1;
        Frame > TargetFrame;
        --Frame)
    {
        const FBacktraceFrame& Record = Backtrace[static_cast<size_t>(Frame)];
        if (Token < 0 || Token >= static_cast<int32>(Record.PreviousToken.size())) break;
        const int32 Previous = Record.PreviousToken[static_cast<size_t>(Token)];
        if (Previous == INDEX_NONE) break;
        Token = Previous;
    }
    return FMath::Clamp(Token, 0, Transcript.NumTokens() - 1);
}

float FOffgridAINeuralStreamingAligner::ConfidenceAt(
    int32 FrameIndex,
    int32 Token) const
{
    if (FrameIndex < 0 || FrameIndex >= static_cast<int32>(Backtrace.size())) return 0.0f;
    const FBacktraceFrame& Record = Backtrace[static_cast<size_t>(FrameIndex)];
    return Token >= 0 && Token < static_cast<int32>(Record.Confidence.size())
        ? Record.Confidence[static_cast<size_t>(Token)]
        : 0.0f;
}

void FOffgridAINeuralStreamingAligner::AssignCommittedFrame(
    int32 Token,
    float CenterSec,
    float Confidence,
    float CommitPlaybackSec,
    FOffgridAICommittedVisemeTrack& InOutTrack,
    FName CommitReason)
{
    Token = FMath::Clamp(Token, 0, Transcript.NumTokens() - 1);
    if (LastAssignedToken != INDEX_NONE && Token != LastAssignedToken)
        FinishToken(LastAssignedToken, CommitPlaybackSec, InOutTrack, CommitReason);
    if (TokenFirstSec[Token] < 0.0f) TokenFirstSec[Token] = CenterSec;
    TokenLastSec[Token] = FMath::Max(TokenLastSec[Token], CenterSec);
    TokenConfidence[Token] = FMath::Max(TokenConfidence[Token], Confidence);
    LastAssignedToken = Token;
}

void FOffgridAINeuralStreamingAligner::FinishToken(
    int32 Token,
    float CommitPlaybackSec,
    FOffgridAICommittedVisemeTrack& InOutTrack,
    FName CommitReason)
{
    if (!Transcript.EventIndices.IsValidIndex(Token)
        || TokenFinished[Token]
        || Transcript.SilenceTokens[Token])
    {
        if (TokenFinished.IsValidIndex(Token)) TokenFinished[Token] = true;
        return;
    }
    const int32 EventIndex = Transcript.EventIndices[Token];
    if (!TextPlan || !TextPlan->Events.IsValidIndex(EventIndex)) return;
    const FOffgridAITextVisemeEvent& Planned = TextPlan->Events[EventIndex];

    const float FirstSec = TokenFirstSec[Token] >= 0.0f
        ? TokenFirstSec[Token]
        : (InOutTrack.Events.Num() > 0
            ? InOutTrack.Events.Last().FinalRenderCenterSeconds + 0.010f
            : 0.0f);
    const float LastSec = FMath::Max(TokenLastSec[Token], FirstSec);
    float CenterSec = 0.5f * (FirstSec + LastSec);
    if (InOutTrack.Events.Num() > 0)
        CenterSec = FMath::Max(
            CenterSec,
            InOutTrack.Events.Last().FinalRenderCenterSeconds + 0.00001f);
    // A path revision can occasionally finish a tail token after its entire
    // predicted presentation window is already behind playback. Preserve its
    // neural identity and order, but move only these otherwise-undeliverable
    // outliers to the live edge so the articulation is actually observable.
    if (CommitPlaybackSec - CenterSec > 0.200f)
        CenterSec = FMath::Max(CenterSec, CommitPlaybackSec + 0.020f);

    FOffgridAICommittedVisemeEvent Event;
    Event.EventIndex = EventIndex;
    Event.PoseID = Planned.PoseID;
    Event.Strength = Planned.Strength;
    Event.SourceWord = Planned.SourceText;
    Event.WordIndex = Planned.WordIndex;
    Event.SpeechRegionIndex = RegionForTime(CenterSec);
    Event.SentenceIndex = Planned.SentenceIndex;
    Event.bIsStrongVisibleEvent = Planned.bIsStrongVisibleEvent;
    Event.bIsRenderable = Planned.bIsRenderable;
    Event.TextCenterNorm = Planned.StartNorm
        + 0.5f * (Planned.EndNorm - Planned.StartNorm);
    Event.TextDiagnosticCenterSeconds = CenterSec;
    Event.FinalRenderCenterSeconds = CenterSec;
    // Preserve the neural center and ordering, but make every selected visible
    // state observable for at least two typical 30 Hz game frames. Without
    // this envelope, a valid 10 ms state can fall entirely between Unreal
    // ticks and appear as missing articulation.
    constexpr float MinimumVisibleEnvelopeSec = 0.060f;
    const bool bWordFinalVisibleToken =
        Token + 1 < Transcript.NumTokens()
        && Transcript.SilenceTokens.IsValidIndex(Token + 1)
        && Transcript.SilenceTokens[Token + 1];
    const float ReleaseTailSec = bWordFinalVisibleToken && bInSpeech
        ? FMath::Clamp(DurationPriors[Token] * 1.25f, 0.030f, 0.100f)
        : 0.5f * MinimumVisibleEnvelopeSec;
    Event.RenderStartSeconds = FMath::Max(
        0.0f,
        FMath::Min(FirstSec - 0.005f,
            CenterSec - 0.5f * MinimumVisibleEnvelopeSec));
    const float BaseRenderEndSec = FMath::Max(
        FMath::Max(Event.RenderStartSeconds + 0.010f, LastSec + 0.005f),
        CenterSec + 0.5f * MinimumVisibleEnvelopeSec);
    Event.RenderEndSeconds = FMath::Max(
        BaseRenderEndSec,
        CenterSec + ReleaseTailSec);
    Event.PriorStartSeconds = Event.RenderStartSeconds;
    Event.PriorCenterSeconds = CenterSec;
    Event.PriorEndSeconds = Event.RenderEndSeconds;
    Event.LeadAdjustedCenterSeconds = CenterSec;
    Event.SourcePhoneIndex = Planned.SourcePhoneGlobalIndex;
    Event.SourcePhoneBase = Planned.SourcePhoneBase;
    Event.SourcePhoneClass = NeuralPhoneClass(Planned.SourcePhoneBase);
    Event.bMappedToObservedSpeech = Event.SpeechRegionIndex != INDEX_NONE;
    Event.CommitPlaybackSeconds = CommitPlaybackSec;
    Event.CommitLeadSeconds = CenterSec - CommitPlaybackSec;
    Event.CommitReason = CommitReason;
    Event.AcousticAnchorKind = FName(TEXT("neural_monotonic_state"));
    Event.AcousticAnchorSeconds = CenterSec;
    Event.AcousticAnchorErrorSeconds = 0.0f;
    InOutTrack.Events.Add(Event);
    TokenFinished[Token] = true;
    bHasCommittedTrack = true;
}

int32 FOffgridAINeuralStreamingAligner::RegionForTime(float CenterSec) const
{
    int32 BestRegion = INDEX_NONE;
    float BestDistance = std::numeric_limits<float>::infinity();
    for (const FOffgridAIStreamingSpeechRegion& Region : SpeechRegions)
    {
        const float EndSec = Region.bEnded
            ? Region.AudioBufferEndSec
            : FMath::Max(Region.AudioBufferEndSec, Region.AudioBufferLastSpeechSec);
        const float Distance = CenterSec < Region.AudioBufferStartSec
            ? Region.AudioBufferStartSec - CenterSec
            : (CenterSec > EndSec ? CenterSec - EndSec : 0.0f);
        if (Distance < BestDistance)
        {
            BestDistance = Distance;
            BestRegion = Region.SpeechRegionIndex;
        }
    }
    return BestRegion;
}

void FOffgridAINeuralStreamingAligner::RefreshTrackRegions(
    FOffgridAICommittedVisemeTrack& InOutTrack) const
{
    InOutTrack.SpeechRegions.Reset();
    for (const FOffgridAIStreamingSpeechRegion& Region : SpeechRegions)
    {
        FOffgridAICommittedVisemeTrack::FSpeechRegion TrackRegion;
        TrackRegion.SpeechRegionIndex = Region.SpeechRegionIndex;
        TrackRegion.StartSeconds = Region.AudioBufferStartSec;
        TrackRegion.EndSeconds = FMath::Max(
            Region.AudioBufferStartSec,
            Region.bEnded ? Region.AudioBufferEndSec : Region.AudioBufferLastSpeechSec);
        TrackRegion.bEnded = Region.bEnded;
        InOutTrack.SpeechRegions.Add(TrackRegion);
    }
    if (InOutTrack.SpeechRegions.Num() > 0)
    {
        InOutTrack.SpeechStartSeconds = InOutTrack.SpeechRegions[0].StartSeconds;
        InOutTrack.SpeechEndSeconds = InOutTrack.SpeechRegions.Last().EndSeconds;
    }
}

void FOffgridAINeuralStreamingAligner::Finalize(
    float ObservedAudioEndSec,
    float CommitPlaybackSec,
    FOffgridAICommittedVisemeTrack& InOutTrack)
{
    if (bFinalized || !TextPlan) return;
    bFinalized = true;
    if (bInSpeech && SpeechRegions.Num() > 0)
    {
        FOffgridAIStreamingSpeechRegion& Region = SpeechRegions.Last();
        Region.AudioBufferEndSec = FMath::Max(
            Region.AudioBufferStartSec,
            Region.AudioBufferLastSpeechSec);
        Region.ProvisionalEndSec = Region.AudioBufferEndSec;
        Region.EndDecisionSec = FMath::Max(ObservedAudioEndSec, Region.AudioBufferEndSec);
        Region.bEnded = true;
        Region.EndReason = FName(TEXT("neural_end_of_stream"));
        bInSpeech = false;
    }

    const int32 FrameCount = static_cast<int32>(Backtrace.size());
    if (FrameCount > 0)
    {
        std::vector<int32> Tail(static_cast<size_t>(FrameCount), 0);
        int32 State = BestCurrentToken();
        for (int32 Frame = FrameCount - 1; Frame >= NextCommitFrame; --Frame)
        {
            Tail[static_cast<size_t>(Frame)] = State;
            const FBacktraceFrame& Record = Backtrace[static_cast<size_t>(Frame)];
            if (Frame > NextCommitFrame
                && State >= 0
                && State < static_cast<int32>(Record.PreviousToken.size())
                && Record.PreviousToken[static_cast<size_t>(State)] != INDEX_NONE)
                State = Record.PreviousToken[static_cast<size_t>(State)];
        }

        const int32 Prefix = FMath::Max(LastAssignedToken, 0);
        const int32 TailFrames = FrameCount - NextCommitFrame;
        const int32 Remaining = Transcript.NumTokens() - 1 - Prefix;
        if (TailFrames > 0 && Tail.back() < Transcript.NumTokens() - 1
            && Remaining <= TailFrames)
        {
            for (int32 Offset = 0; Offset < TailFrames; ++Offset)
            {
                const int32 Completed = FMath::Min(
                    Remaining,
                    static_cast<int32>(std::floor(
                        (Offset + 1.0) * Remaining / TailFrames)));
                Tail[static_cast<size_t>(NextCommitFrame + Offset)] = Prefix + Completed;
            }
        }

        for (int32 Frame = NextCommitFrame; Frame < FrameCount; ++Frame)
        {
            int32 Token = Tail[static_cast<size_t>(Frame)];
            if (LastAssignedToken >= 0)
                Token = FMath::Clamp(Token, LastAssignedToken, LastAssignedToken + 1);
            AssignCommittedFrame(
                Token,
                Backtrace[static_cast<size_t>(Frame)].CenterSec,
                ConfidenceAt(Frame, Token),
                CommitPlaybackSec,
                InOutTrack,
                FName(TEXT("neural_end_of_stream")));
        }
    }

    float SyntheticCenter = FrameCount > 0
        ? Backtrace.back().CenterSec
        : FMath::Max(ObservedAudioEndSec, 0.0f);
    while (LastAssignedToken + 1 < Transcript.NumTokens())
    {
        SyntheticCenter += 0.010f;
        AssignCommittedFrame(
            LastAssignedToken + 1,
            SyntheticCenter,
            0.0f,
            CommitPlaybackSec,
            InOutTrack,
            FName(TEXT("neural_end_of_stream_completion")));
    }
    if (LastAssignedToken >= 0)
        FinishToken(
            LastAssignedToken,
            CommitPlaybackSec,
            InOutTrack,
            FName(TEXT("neural_end_of_stream")));
    RefreshTrackRegions(InOutTrack);
}
