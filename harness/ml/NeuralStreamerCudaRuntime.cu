#include "NeuralStreamerCudaRuntime.h"
#include "NeuralStreamerModelFormat.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

namespace offgridai::neural_streamer {
namespace {

constexpr std::size_t kMeanOffset = 0;
constexpr std::size_t kScaleOffset = kMeanOffset + kAudioMeanCount;
constexpr std::size_t kConv1WeightOffset = kScaleOffset + kAudioScaleCount;
constexpr std::size_t kConv1BiasOffset = kConv1WeightOffset + kAudioConv1WeightCount;
constexpr std::size_t kConv2WeightOffset = kConv1BiasOffset + kAudioConv1BiasCount;
constexpr std::size_t kConv2BiasOffset = kConv2WeightOffset + kAudioConv2WeightCount;
constexpr std::size_t kToken1WeightOffset = kConv2BiasOffset + kAudioConv2BiasCount;
constexpr std::size_t kToken1BiasOffset = kToken1WeightOffset + kTokenLinear1WeightCount;
constexpr std::size_t kToken2WeightOffset = kToken1BiasOffset + kTokenLinear1BiasCount;
constexpr std::size_t kToken2BiasOffset = kToken2WeightOffset + kTokenLinear2WeightCount;
constexpr std::size_t kRegionConv1WeightOffset = kToken2BiasOffset + kTokenLinear2BiasCount;
constexpr std::size_t kRegionConv1BiasOffset = kRegionConv1WeightOffset + kRegionConv1WeightCount;
constexpr std::size_t kRegionConv2WeightOffset = kRegionConv1BiasOffset + kRegionConv1BiasCount;
constexpr std::size_t kRegionConv2BiasOffset = kRegionConv2WeightOffset + kRegionConv2WeightCount;
constexpr std::size_t kRegionWeightOffset = kRegionConv2BiasOffset + kRegionConv2BiasCount;
constexpr std::size_t kRegionBiasOffset = kRegionWeightOffset + kRegionLinearWeightCount;
static_assert(kRegionBiasOffset + kRegionLinearBiasCount == kCheckpointFloatCount);

static std::string cuda_error(const char* operation, cudaError_t error)
{
    std::ostringstream out;
    out << operation << ": " << cudaGetErrorString(error);
    return out.str();
}

__global__ void encode_tokens_kernel(
    const float* features,
    int token_count,
    const __half* weights,
    __half* embeddings,
    float* silence_mask)
{
    const int token = blockIdx.x;
    const int hidden = threadIdx.x;
    if (token >= token_count || hidden >= kHiddenDimensions) return;
    __shared__ float first[kHiddenDimensions];

    float value = __half2float(weights[kToken1BiasOffset + hidden]);
    const __half* row = weights + kToken1WeightOffset + hidden * kTokenDimensions;
    const float* token_features = features + token * kTokenDimensions;
    for (int feature = 0; feature < kTokenDimensions; ++feature)
        value += __half2float(row[feature]) * token_features[feature];
    first[hidden] = fmaxf(0.0f, value);
    __syncthreads();

    value = __half2float(weights[kToken2BiasOffset + hidden]);
    row = weights + kToken2WeightOffset + hidden * kHiddenDimensions;
    for (int input = 0; input < kHiddenDimensions; ++input)
        value += __half2float(row[input]) * first[input];
    embeddings[token * kHiddenDimensions + hidden] = __float2half(value);
    if (hidden == 0)
        silence_mask[token] = token_features[kPhoneBuckets + kPoseBuckets + 7];
}

__global__ void score_chunk_kernel(
    const float* audio,
    int context_frame_count,
    int frame_count,
    const __half* weights,
    const __half* token_embeddings,
    const float* token_silence_mask,
    int active_token,
    int scored_tokens,
    float* scores,
    float* region_logits)
{
    const int hidden = threadIdx.x;
    if (hidden >= kHiddenDimensions) return;
    constexpr int kWindowFrames = kCausalContextFrames + kRuntimeChunkFrames;
    __shared__ float conv1[kWindowFrames * kHiddenDimensions];
    __shared__ float conv2[kRuntimeChunkFrames * kHiddenDimensions];
    __shared__ float region_conv1[kWindowFrames * kHiddenDimensions];
    __shared__ float region_conv2[kRuntimeChunkFrames * kHiddenDimensions];
    __shared__ float reduction[kHiddenDimensions];
    __shared__ float region_score[kRuntimeChunkFrames];
    const int total_frames = context_frame_count + frame_count;

    for (int frame = 0; frame < total_frames; ++frame) {
        float value = __half2float(weights[kConv1BiasOffset + hidden]);
        float region_value = __half2float(weights[kRegionConv1BiasOffset + hidden]);
        const __half* output_weights = weights + kConv1WeightOffset
            + hidden * kAudioFeatureCount * kConvolutionKernel;
        const __half* region_output_weights = weights + kRegionConv1WeightOffset
            + hidden * kAudioFeatureCount * kConvolutionKernel;
        for (int input = 0; input < kAudioFeatureCount; ++input) {
            const float mean = __half2float(weights[kMeanOffset + input]);
            const float scale = __half2float(weights[kScaleOffset + input]);
            for (int kernel = 0; kernel < kConvolutionKernel; ++kernel) {
                const int source_frame = frame
                    + kernel * kAudioConv1Dilation
                    - (kConvolutionKernel - 1) * kAudioConv1Dilation;
                if (source_frame < 0) continue;
                const float normalized =
                    (audio[source_frame * kAudioFeatureCount + input] - mean) / scale;
                value += __half2float(output_weights[input * kConvolutionKernel + kernel])
                    * normalized;
                region_value += __half2float(
                    region_output_weights[input * kConvolutionKernel + kernel]) * normalized;
            }
        }
        conv1[frame * kHiddenDimensions + hidden] = fmaxf(0.0f, value);
        region_conv1[frame * kHiddenDimensions + hidden] = fmaxf(0.0f, region_value);
    }
    __syncthreads();

    for (int output_frame = 0; output_frame < frame_count; ++output_frame) {
        const int frame = context_frame_count + output_frame;
        float value = __half2float(weights[kConv2BiasOffset + hidden]);
        float region_value = __half2float(weights[kRegionConv2BiasOffset + hidden]);
        const __half* output_weights = weights + kConv2WeightOffset
            + hidden * kHiddenDimensions * kConvolutionKernel;
        const __half* region_output_weights = weights + kRegionConv2WeightOffset
            + hidden * kHiddenDimensions * kConvolutionKernel;
        for (int input = 0; input < kHiddenDimensions; ++input) {
            for (int kernel = 0; kernel < kConvolutionKernel; ++kernel) {
                const int source_frame = frame
                    + kernel * kAudioConv2Dilation
                    - (kConvolutionKernel - 1) * kAudioConv2Dilation;
                if (source_frame < 0) continue;
                value += __half2float(output_weights[input * kConvolutionKernel + kernel])
                    * conv1[source_frame * kHiddenDimensions + input];
            }
            for (int kernel = 0; kernel < kConvolutionKernel; ++kernel) {
                const int source_frame = frame
                    + kernel * kRegionConv2Dilation
                    - (kConvolutionKernel - 1) * kRegionConv2Dilation;
                if (source_frame < 0) continue;
                region_value += __half2float(
                    region_output_weights[input * kConvolutionKernel + kernel])
                    * region_conv1[source_frame * kHiddenDimensions + input];
            }
        }
        conv2[output_frame * kHiddenDimensions + hidden] = fmaxf(0.0f, value);
        region_conv2[output_frame * kHiddenDimensions + hidden] = fmaxf(0.0f, region_value);
    }
    __syncthreads();

    for (int frame = 0; frame < frame_count; ++frame) {
        reduction[hidden] = region_conv2[frame * kHiddenDimensions + hidden]
            * __half2float(weights[kRegionWeightOffset + hidden]);
        __syncthreads();
        for (int stride = kHiddenDimensions / 2; stride > 0; stride /= 2) {
            if (hidden < stride) reduction[hidden] += reduction[hidden + stride];
            __syncthreads();
        }
        if (hidden == 0) {
            region_score[frame] = reduction[0] + __half2float(weights[kRegionBiasOffset]);
            region_logits[frame] = region_score[frame];
        }
        __syncthreads();
        for (int token = 0; token < scored_tokens; ++token) {
            reduction[hidden] = conv2[frame * kHiddenDimensions + hidden]
                * __half2float(token_embeddings[
                    (active_token + token) * kHiddenDimensions + hidden]);
            __syncthreads();
            for (int stride = kHiddenDimensions / 2; stride > 0; stride /= 2) {
                if (hidden < stride) reduction[hidden] += reduction[hidden + stride];
                __syncthreads();
            }
            if (hidden == 0)
                scores[frame * kForwardTokenWindow + token] = reduction[0] / 8.0f
                    + 2.0f * tanhf(region_score[frame])
                    * (1.0f - 2.0f * token_silence_mask[active_token + token]);
            __syncthreads();
        }
    }
}

}  // namespace

struct CudaRuntime::Impl {
    cudaStream_t Stream = nullptr;
    __half* DeviceWeights = nullptr;
    __half* DeviceTokenEmbeddings = nullptr;
    float* DeviceTokenSilenceMask = nullptr;
    float* DeviceTokenFeatures = nullptr;
    float* DeviceAudio = nullptr;
    float* DeviceScores = nullptr;
    float* DeviceRegionLogits = nullptr;
    float* HostAudio = nullptr;
    float* HostScores = nullptr;
    float* HostRegionLogits = nullptr;
    float* HostTokenFeatures = nullptr;
    int TokenCapacity = 0;
    int TokenCount = 0;
    int StreamPriority = 0;
    int ContextFrameCount = 0;
    std::size_t CheckpointBytes = 0;
    std::array<float, kCausalContextFrames * kAudioFeatureCount> Context{};
    std::string Error;

    Impl()
    {
        int least_priority = 0;
        int greatest_priority = 0;
        if (cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority) == cudaSuccess)
            StreamPriority = least_priority;
        if (cudaStreamCreateWithPriority(&Stream, cudaStreamNonBlocking, StreamPriority) != cudaSuccess)
            Stream = nullptr;
        cudaMallocHost(&HostAudio,
            (kCausalContextFrames + kRuntimeChunkFrames) * kAudioFeatureCount * sizeof(float));
        cudaMallocHost(&HostScores,
            kRuntimeChunkFrames * kForwardTokenWindow * sizeof(float));
        cudaMallocHost(&HostRegionLogits, kRuntimeChunkFrames * sizeof(float));
        cudaMalloc(&DeviceAudio,
            (kCausalContextFrames + kRuntimeChunkFrames) * kAudioFeatureCount * sizeof(float));
        cudaMalloc(&DeviceScores,
            kRuntimeChunkFrames * kForwardTokenWindow * sizeof(float));
        cudaMalloc(&DeviceRegionLogits, kRuntimeChunkFrames * sizeof(float));
    }

    ~Impl()
    {
        if (Stream) cudaStreamSynchronize(Stream);
        if (DeviceWeights) cudaFree(DeviceWeights);
        if (DeviceTokenEmbeddings) cudaFree(DeviceTokenEmbeddings);
        if (DeviceTokenSilenceMask) cudaFree(DeviceTokenSilenceMask);
        if (DeviceTokenFeatures) cudaFree(DeviceTokenFeatures);
        if (DeviceAudio) cudaFree(DeviceAudio);
        if (DeviceScores) cudaFree(DeviceScores);
        if (DeviceRegionLogits) cudaFree(DeviceRegionLogits);
        if (HostAudio) cudaFreeHost(HostAudio);
        if (HostScores) cudaFreeHost(HostScores);
        if (HostRegionLogits) cudaFreeHost(HostRegionLogits);
        if (HostTokenFeatures) cudaFreeHost(HostTokenFeatures);
        if (Stream) cudaStreamDestroy(Stream);
    }

    bool fail(const std::string& message)
    {
        Error = message;
        return false;
    }

    bool reserve_tokens(int count)
    {
        if (count <= TokenCapacity) return true;
        if (DeviceTokenEmbeddings) cudaFree(DeviceTokenEmbeddings);
        if (DeviceTokenSilenceMask) cudaFree(DeviceTokenSilenceMask);
        if (DeviceTokenFeatures) cudaFree(DeviceTokenFeatures);
        if (HostTokenFeatures) cudaFreeHost(HostTokenFeatures);
        DeviceTokenEmbeddings = nullptr;
        DeviceTokenSilenceMask = nullptr;
        DeviceTokenFeatures = nullptr;
        HostTokenFeatures = nullptr;
        TokenCapacity = std::max(count, 256);
        const std::size_t feature_bytes =
            static_cast<std::size_t>(TokenCapacity) * kTokenDimensions * sizeof(float);
        const std::size_t embedding_bytes =
            static_cast<std::size_t>(TokenCapacity) * kHiddenDimensions * sizeof(__half);
        cudaError_t error = cudaMallocHost(&HostTokenFeatures, feature_bytes);
        if (error != cudaSuccess) return fail(cuda_error("cudaMallocHost tokens", error));
        error = cudaMalloc(&DeviceTokenFeatures, feature_bytes);
        if (error != cudaSuccess) return fail(cuda_error("cudaMalloc token features", error));
        error = cudaMalloc(&DeviceTokenEmbeddings, embedding_bytes);
        if (error != cudaSuccess) return fail(cuda_error("cudaMalloc token embeddings", error));
        error = cudaMalloc(&DeviceTokenSilenceMask,
            static_cast<std::size_t>(TokenCapacity) * sizeof(float));
        if (error != cudaSuccess) return fail(cuda_error("cudaMalloc token silence mask", error));
        return true;
    }
};

CudaRuntime::CudaRuntime() : State(std::make_unique<Impl>()) {}
CudaRuntime::~CudaRuntime() = default;

bool CudaRuntime::LoadCheckpoint(const std::string& path)
{
    State->Error.clear();
    if (!State->Stream || !State->HostAudio || !State->HostScores
        || !State->HostRegionLogits || !State->DeviceAudio || !State->DeviceScores
        || !State->DeviceRegionLogits)
        return State->fail("CUDA runtime initialization failed");
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) return State->fail("cannot open checkpoint: " + path);
    State->CheckpointBytes = static_cast<std::size_t>(input.tellg());
    input.seekg(0);
    CheckpointHeader header;
    input.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!input || header.Magic != kCheckpointMagic || header.Version != kCheckpointVersion
        || header.AudioFeatures != kAudioFeatureCount
        || header.TokenFeatures != kTokenDimensions
        || header.HiddenDimensions != kHiddenDimensions
        || header.ParameterCount != kCheckpointFloatCount)
        return State->fail("checkpoint schema mismatch");
    std::vector<float> values(kCheckpointFloatCount);
    input.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(float));
    if (!input) return State->fail("truncated checkpoint");
    std::vector<__half> half_values(values.size());
    std::transform(values.begin(), values.end(), half_values.begin(),
        [](float value) { return __float2half(value); });
    if (State->DeviceWeights) cudaFree(State->DeviceWeights);
    cudaError_t error = cudaMalloc(&State->DeviceWeights, half_values.size() * sizeof(__half));
    if (error != cudaSuccess) return State->fail(cuda_error("cudaMalloc weights", error));
    error = cudaMemcpyAsync(State->DeviceWeights, half_values.data(),
        half_values.size() * sizeof(__half), cudaMemcpyHostToDevice, State->Stream);
    if (error != cudaSuccess) return State->fail(cuda_error("copy weights", error));
    error = cudaStreamSynchronize(State->Stream);
    if (error != cudaSuccess) return State->fail(cuda_error("synchronize weights", error));
    return true;
}

bool CudaRuntime::ResetTokens(const float* features, int token_count)
{
    State->Error.clear();
    if (!State->DeviceWeights) return State->fail("checkpoint is not loaded");
    if (!features || token_count <= 0) return State->fail("token sequence is empty");
    if (!State->reserve_tokens(token_count)) return false;
    State->TokenCount = token_count;
    State->ContextFrameCount = 0;
    State->Context.fill(0.0f);
    const std::size_t bytes = static_cast<std::size_t>(token_count)
        * kTokenDimensions * sizeof(float);
    std::memcpy(State->HostTokenFeatures, features, bytes);
    cudaError_t error = cudaMemcpyAsync(State->DeviceTokenFeatures, State->HostTokenFeatures,
        bytes, cudaMemcpyHostToDevice, State->Stream);
    if (error != cudaSuccess) return State->fail(cuda_error("copy token features", error));
    encode_tokens_kernel<<<token_count, kHiddenDimensions, 0, State->Stream>>>(
        State->DeviceTokenFeatures, token_count, State->DeviceWeights,
        State->DeviceTokenEmbeddings, State->DeviceTokenSilenceMask);
    error = cudaGetLastError();
    if (error != cudaSuccess) return State->fail(cuda_error("encode token launch", error));
    error = cudaStreamSynchronize(State->Stream);
    if (error != cudaSuccess) return State->fail(cuda_error("encode tokens", error));
    return true;
}

bool CudaRuntime::PushChunk(
    const float* audio_features,
    int frame_count,
    int active_token,
    float* scores,
    int& scored_token_count,
    float* region_logits)
{
    State->Error.clear();
    scored_token_count = 0;
    if (!audio_features || !scores || frame_count <= 0 || frame_count > kRuntimeChunkFrames)
        return State->fail("invalid audio chunk");
    if (active_token < 0 || active_token >= State->TokenCount)
        return State->fail("active token is out of range");
    scored_token_count = std::min(kForwardTokenWindow, State->TokenCount - active_token);
    const std::size_t context_values = static_cast<std::size_t>(State->ContextFrameCount)
        * kAudioFeatureCount;
    std::memcpy(State->HostAudio, State->Context.data(), context_values * sizeof(float));
    std::memcpy(State->HostAudio + context_values, audio_features,
        static_cast<std::size_t>(frame_count) * kAudioFeatureCount * sizeof(float));
    const std::size_t input_bytes = static_cast<std::size_t>(
        State->ContextFrameCount + frame_count) * kAudioFeatureCount * sizeof(float);
    cudaError_t error = cudaMemcpyAsync(State->DeviceAudio, State->HostAudio,
        input_bytes, cudaMemcpyHostToDevice, State->Stream);
    if (error != cudaSuccess) return State->fail(cuda_error("copy audio", error));
    score_chunk_kernel<<<1, kHiddenDimensions, 0, State->Stream>>>(
        State->DeviceAudio, State->ContextFrameCount, frame_count, State->DeviceWeights,
        State->DeviceTokenEmbeddings, State->DeviceTokenSilenceMask,
        active_token, scored_token_count,
        State->DeviceScores, State->DeviceRegionLogits);
    error = cudaGetLastError();
    if (error != cudaSuccess) return State->fail(cuda_error("score chunk launch", error));
    const std::size_t output_bytes = static_cast<std::size_t>(frame_count)
        * kForwardTokenWindow * sizeof(float);
    error = cudaMemcpyAsync(State->HostScores, State->DeviceScores,
        output_bytes, cudaMemcpyDeviceToHost, State->Stream);
    if (error != cudaSuccess) return State->fail(cuda_error("copy scores", error));
    error = cudaMemcpyAsync(State->HostRegionLogits, State->DeviceRegionLogits,
        static_cast<std::size_t>(frame_count) * sizeof(float),
        cudaMemcpyDeviceToHost, State->Stream);
    if (error != cudaSuccess) return State->fail(cuda_error("copy region logits", error));
    error = cudaStreamSynchronize(State->Stream);
    if (error != cudaSuccess) return State->fail(cuda_error("score chunk", error));
    std::memcpy(scores, State->HostScores, output_bytes);
    if (region_logits)
        std::memcpy(region_logits, State->HostRegionLogits,
            static_cast<std::size_t>(frame_count) * sizeof(float));

    std::array<float, (kCausalContextFrames + kRuntimeChunkFrames) * kAudioFeatureCount> combined{};
    std::memcpy(combined.data(), State->Context.data(), context_values * sizeof(float));
    std::memcpy(combined.data() + context_values, audio_features,
        static_cast<std::size_t>(frame_count) * kAudioFeatureCount * sizeof(float));
    const int total_frames = State->ContextFrameCount + frame_count;
    const int kept_frames = std::min(kCausalContextFrames, total_frames);
    const int first_kept = total_frames - kept_frames;
    std::memcpy(State->Context.data(),
        combined.data() + first_kept * kAudioFeatureCount,
        static_cast<std::size_t>(kept_frames) * kAudioFeatureCount * sizeof(float));
    State->ContextFrameCount = kept_frames;
    return true;
}

const std::string& CudaRuntime::LastError() const { return State->Error; }

RuntimeFootprint CudaRuntime::Footprint() const
{
    RuntimeFootprint result;
    result.CheckpointBytes = State->CheckpointBytes;
    result.WeightBytes = kCheckpointFloatCount * sizeof(__half);
    result.TokenBytes = static_cast<std::size_t>(State->TokenCount)
        * (kHiddenDimensions * sizeof(__half) + sizeof(float));
    result.StreamPriority = State->StreamPriority;
    return result;
}

}  // namespace offgridai::neural_streamer
