#pragma once

#include "CoreMinimal.h"
#include "Lipsync/OffgridAILipsyncTypes.h"
#include "Lipsync/OffgridAINeuralLipsyncBridge.h"
#include "Lipsync/OffgridAINeuralStreamerRuntime.h"
#include "Lipsync/OffgridAIStreamingAudioFeatures.h"
#include "Lipsync/OffgridAITextVisemePlanner.h"

#include <array>
#include <vector>

// Private live decoder for the CUDA score network. It owns the causal beam,
// fixed-lag path commitment, neural speech regions, and conversion back into
// the stable committed-track contract consumed by Line Coach/FaceDriver.
class FOffgridAINeuralStreamingAligner
{
public:
    bool Begin(
        const FOffgridAITextVisemePlan& Plan,
        const FString& CheckpointPath,
        FString& OutError);
    bool ProcessAvailableFrames(
        const TArray<FOffgridAIStreamingAudioFeatureFrame>& Frames,
        float CommitPlaybackSec,
        FOffgridAICommittedVisemeTrack& InOutTrack,
        FString& OutError);
    void Finalize(
        float ObservedAudioEndSec,
        float CommitPlaybackSec,
        FOffgridAICommittedVisemeTrack& InOutTrack);

    const TArray<FOffgridAIStreamingSpeechRegion>& GetSpeechRegions() const
    {
        return SpeechRegions;
    }
    bool HasCommittedTrack() const { return bHasCommittedTrack; }

private:
    // Leave 50 ms of presentation lead inside Line Coach's 350 ms playback
    // reservoir. A 350 ms backtrace was accurate offline but committed most
    // poses after their attack window had already begun at runtime.
    static constexpr int32 FixedLagFrames = 30;
    // Crossing a transcript pause fence is the one irreversible decision that
    // can change observed-region ownership. Wait three additional 10 ms frames
    // while that crossing is unresolved; ordinary commits retain 300 ms lag.
    static constexpr int32 BoundaryCommitLagFrames = 33;
    struct FBacktraceFrame
    {
        float CenterSec = 0.0f;
        std::vector<int32> PreviousToken;
        std::vector<float> Confidence;
    };

    void DecodeFrame(
        const float* Scores,
        int32 ScoredTokenCount,
        float RegionLogit,
        const FOffgridAIStreamingAudioFeatureFrame& Frame,
        float CommitPlaybackSec,
        FOffgridAICommittedVisemeTrack& InOutTrack);
    void DecodeRegion(
        float RegionLogit,
        const FOffgridAIStreamingAudioFeatureFrame& Frame);
    int32 BestCurrentToken() const;
    int32 BacktrackToFrame(int32 Token, int32 TargetFrame) const;
    float ConfidenceAt(int32 FrameIndex, int32 Token) const;
    void AssignCommittedFrame(
        int32 Token,
        float CenterSec,
        float Confidence,
        float CommitPlaybackSec,
        FOffgridAICommittedVisemeTrack& InOutTrack,
        FName CommitReason);
    void FinishToken(
        int32 Token,
        float CommitPlaybackSec,
        FOffgridAICommittedVisemeTrack& InOutTrack,
        FName CommitReason);
    void RefreshTrackRegions(FOffgridAICommittedVisemeTrack& InOutTrack) const;
    int32 RegionForTime(float CenterSec) const;

    offgridai::neural_streamer::CudaRuntime Runtime;
    const FOffgridAITextVisemePlan* TextPlan = nullptr;
    FOffgridAINeuralTranscriptTensor Transcript;
    TArray<float> DurationPriors;
    TArray<float> TokenFirstSec;
    TArray<float> TokenLastSec;
    TArray<float> TokenConfidence;
    TArray<bool> TokenFinished;
    std::vector<float> PreviousCosts;
    std::vector<float> CurrentCosts;
    std::vector<int32> PreviousDwells;
    std::vector<int32> CurrentDwells;
    std::vector<FBacktraceFrame> Backtrace;
    int32 NextInputFrame = 0;
    int32 NextCommitFrame = 0;
    int32 LastAssignedToken = INDEX_NONE;
    int32 PendingBoundaryCommitFrame = INDEX_NONE;
    int32 PendingBoundaryState = INDEX_NONE;
    int32 PendingBoundaryPauseToken = INDEX_NONE;
    int32 ResolvedBoundaryPauseToken = INDEX_NONE;
    bool bHasCommittedTrack = false;
    bool bFinalized = false;
    bool bCheckpointLoaded = false;
    FString LoadedCheckpointPath;

    TArray<FOffgridAIStreamingSpeechRegion> SpeechRegions;
    int32 SpeechCandidateStartFrame = INDEX_NONE;
    float SpeechCandidateStartSec = 0.0f;
    int32 QuietCandidateStartFrame = INDEX_NONE;
    float QuietCandidateStartSec = 0.0f;
    bool bInSpeech = false;
};
