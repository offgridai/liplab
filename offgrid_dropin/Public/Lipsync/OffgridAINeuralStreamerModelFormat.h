#pragma once

#include <cstddef>
#include <cstdint>

namespace offgridai::neural_streamer {

constexpr std::uint32_t kCheckpointMagic = 0x4c504e53u;  // "SNPL"
constexpr std::uint32_t kCheckpointVersion = 5;
constexpr int kAudioFeatureCount = 18;
constexpr int kPhoneBuckets = 64;
constexpr int kPoseBuckets = 32;
constexpr int kLegacyTokenContinuous = 11;
constexpr int kPunctuationTypeCount = 8;
constexpr int kPunctuationFeatureCount = 1 + kPunctuationTypeCount;
constexpr int kTokenContinuous = kLegacyTokenContinuous + kPunctuationFeatureCount;
constexpr int kTokenDimensions = kPhoneBuckets + kPoseBuckets + kTokenContinuous;
constexpr int kHiddenDimensions = 64;
constexpr int kConvolutionKernel = 3;
constexpr int kAudioConv1Dilation = 1;
constexpr int kAudioConv2Dilation = 1;
constexpr int kRegionConv1Dilation = 1;
constexpr int kRegionConv2Dilation = 4;
constexpr int kCausalContextFrames =
    (kConvolutionKernel - 1) * (kRegionConv1Dilation + kRegionConv2Dilation);
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
constexpr std::size_t kRegionConv1WeightCount = kAudioConv1WeightCount;
constexpr std::size_t kRegionConv1BiasCount = kAudioConv1BiasCount;
constexpr std::size_t kRegionConv2WeightCount = kAudioConv2WeightCount;
constexpr std::size_t kRegionConv2BiasCount = kAudioConv2BiasCount;
constexpr std::size_t kRegionLinearWeightCount = kHiddenDimensions;
constexpr std::size_t kRegionLinearBiasCount = 1;
constexpr std::size_t kCheckpointFloatCount =
    kAudioMeanCount + kAudioScaleCount
    + kAudioConv1WeightCount + kAudioConv1BiasCount
    + kAudioConv2WeightCount + kAudioConv2BiasCount
    + kTokenLinear1WeightCount + kTokenLinear1BiasCount
    + kTokenLinear2WeightCount + kTokenLinear2BiasCount
    + kRegionConv1WeightCount + kRegionConv1BiasCount
    + kRegionConv2WeightCount + kRegionConv2BiasCount
    + kRegionLinearWeightCount + kRegionLinearBiasCount;

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
