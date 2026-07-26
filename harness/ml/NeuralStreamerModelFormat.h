#pragma once

#include <cstddef>
#include <cstdint>

namespace offgridai::neural_streamer {

constexpr std::uint32_t kCheckpointMagic = 0x4c504e53u;  // "SNPL"
constexpr std::uint32_t kCheckpointVersion = 1;
constexpr int kAudioFeatureCount = 20;
constexpr int kPhoneBuckets = 64;
constexpr int kPoseBuckets = 32;
constexpr int kTokenContinuous = 10;
constexpr int kTokenDimensions = kPhoneBuckets + kPoseBuckets + kTokenContinuous;
constexpr int kHiddenDimensions = 64;
constexpr int kConvolutionKernel = 3;
constexpr int kCausalContextFrames = 4;
constexpr int kRuntimeChunkFrames = 4;
constexpr int kForwardTokenWindow = 4;

constexpr std::size_t kAudioMeanCount = kAudioFeatureCount;
constexpr std::size_t kAudioScaleCount = kAudioFeatureCount;
constexpr std::size_t kAudioConv1WeightCount =
    kHiddenDimensions * kAudioFeatureCount * kConvolutionKernel;
constexpr std::size_t kAudioConv1BiasCount = kHiddenDimensions;
constexpr std::size_t kAudioConv2WeightCount =
    kHiddenDimensions * kHiddenDimensions * kConvolutionKernel;
constexpr std::size_t kAudioConv2BiasCount = kHiddenDimensions;
constexpr std::size_t kTokenLinear1WeightCount = kHiddenDimensions * kTokenDimensions;
constexpr std::size_t kTokenLinear1BiasCount = kHiddenDimensions;
constexpr std::size_t kTokenLinear2WeightCount = kHiddenDimensions * kHiddenDimensions;
constexpr std::size_t kTokenLinear2BiasCount = kHiddenDimensions;
constexpr std::size_t kCheckpointFloatCount =
    kAudioMeanCount + kAudioScaleCount
    + kAudioConv1WeightCount + kAudioConv1BiasCount
    + kAudioConv2WeightCount + kAudioConv2BiasCount
    + kTokenLinear1WeightCount + kTokenLinear1BiasCount
    + kTokenLinear2WeightCount + kTokenLinear2BiasCount;

struct CheckpointHeader {
    std::uint32_t Magic = kCheckpointMagic;
    std::uint32_t Version = kCheckpointVersion;
    std::uint32_t AudioFeatures = kAudioFeatureCount;
    std::uint32_t TokenFeatures = kTokenDimensions;
    std::uint32_t HiddenDimensions = kHiddenDimensions;
    std::uint32_t ParameterCount = static_cast<std::uint32_t>(kCheckpointFloatCount);
};

static_assert(sizeof(CheckpointHeader) == 24);

}  // namespace offgridai::neural_streamer
