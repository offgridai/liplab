#pragma once

#include <array>
#include <cstdint>

namespace offgridai::timing_ml {

constexpr int kContractVersion = 1;
constexpr int kAudioFeatureCount = 20;
constexpr int kAudioHistoryFrames = 32;

// Fixed-shape, causal input intended for a small CUDA model. It is evaluated
// once after the deterministic scheduler has formed a complete event proposal
// and immediately before that event becomes immutable. Transcript phone and
// pose IDs are dictionary indices prepared when the line begins; audio history
// contains only frames whose end time is no later than AudioCutoffSec.
struct TimingModelInput {
    std::array<float, kAudioFeatureCount * kAudioHistoryFrames> AudioHistory{};
    std::int32_t ValidAudioFrames = 0;

    std::int32_t NextEventIndex = -1;
    std::int32_t NextPhoneIndex = -1;
    std::int32_t WordIndex = -1;
    std::int32_t SpeechRegionIndex = -1;
    std::int32_t PhoneDictionaryId = 0;
    std::int32_t PoseDictionaryId = 0;

    float PlaybackSec = 0.0f;
    float AudioCutoffSec = 0.0f;
    float CandidateCenterSec = 0.0f;
    float CommitFrontierSec = 0.0f;
    float CommitLeadSec = 0.0f;
    float TimelineRate = 1.0f;
    float PhoneDurationPriorSec = 0.0f;
    float ActiveRegionStartSec = -1.0f;
    float ActiveRegionEndSec = -1.0f;
    float LastMatchedPhoneAudioSec = -1.0f;
    float LastMatchedConfidence = 0.0f;
};

// The model never returns a viseme, phone, word, or event ordering decision.
// It may only advise a bounded correction for the scheduler's next
// transcript-owned event. The deterministic integrator owns confidence
// fallback, commit immutability, monotonicity, and speech-region containment.
struct TimingAdvice {
    float CenterDeltaSec = 0.0f;
    float Confidence = 0.0f;
};

class IStreamingTimingModel {
public:
    virtual ~IStreamingTimingModel() = default;
    virtual void Reset() = 0;
    virtual TimingAdvice Evaluate(const TimingModelInput& input) = 0;
};

}  // namespace offgridai::timing_ml
