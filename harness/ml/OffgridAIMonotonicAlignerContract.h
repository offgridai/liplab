#pragma once

#include "NeuralStreamerModelFormat.h"

#include <array>
#include <cstdint>
#include <span>

namespace offgridai::monotonic_ml {

constexpr int kContractVersion = 1;
constexpr int kForwardTokenWindow = neural_streamer::kForwardTokenWindow;
constexpr int kAudioFeatureCount = neural_streamer::kAudioFeatureCount;

// Constructed from the transcript before audio begins. Identity and order are
// immutable; inference decides only which token is active and when it advances.
struct TranscriptToken {
    std::int32_t PhoneDictionaryId = 0;
    std::int32_t PoseDictionaryId = 0;
    std::int32_t WordIndex = -1;
    std::int32_t PhoneIndexInWord = -1;
    float DurationPriorSec = 0.0f;
    float VisualWeight = 1.0f;
    float IsVowel = 0.0f;
    float IsPermittedSilence = 0.0f;
};

struct CausalFrame {
    std::array<float, kAudioFeatureCount> Audio{};
    float AudioStartSec = 0.0f;
    float AudioEndSec = 0.0f;
    float PlaybackSec = 0.0f;
};

struct AlignmentDecision {
    std::int32_t ActiveTokenIndex = -1;
    std::int32_t AdvancedTokenCount = 0;
    std::int32_t SpeechRegionIndex = -1;
    std::int32_t ActiveWordIndex = -1;
    std::int32_t ActiveSyllableNucleusPhoneIndex = -1;
    float TransitionAudioSec = -1.0f;
    float PathConfidence = 0.0f;
    bool InSpeech = false;
    bool StartedSpeechRegion = false;
    bool EndedSpeechRegion = false;
    bool ResumedSpeech = false;
    bool StartedWord = false;
    bool ReachedSyllableNucleus = false;
    bool UseDeterministicFallback = false;
};

// Implementations own the persistent forward/Viterbi state. PushFrame is
// causal and may advance only through consecutive transcript tokens.
class IStreamingMonotonicAligner {
public:
    virtual ~IStreamingMonotonicAligner() = default;
    virtual void Reset(std::span<const TranscriptToken> transcript) = 0;
    virtual AlignmentDecision PushFrame(const CausalFrame& frame) = 0;
};

}  // namespace offgridai::monotonic_ml
