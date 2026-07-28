#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace offgridai::neural_streamer {

struct RuntimeFootprint {
    std::size_t CheckpointBytes = 0;
    std::size_t WeightBytes = 0;
    std::size_t TokenBytes = 0;
    int StreamPriority = 0;
    std::uint64_t CheckpointFingerprint = 0;
};

struct RuntimeDeviceInfo {
    std::string Name;
    int DeviceIndex = -1;
    int ComputeCapabilityMajor = 0;
    int ComputeCapabilityMinor = 0;
    int DriverVersion = 0;
    int RuntimeVersion = 0;
};

class CudaRuntime {
public:
    CudaRuntime();
    ~CudaRuntime();
    CudaRuntime(const CudaRuntime&) = delete;
    CudaRuntime& operator=(const CudaRuntime&) = delete;

    bool LoadCheckpoint(const std::string& path);
    bool LoadCheckpoint(const void* data, std::size_t size);
    bool ResetTokens(const float* features, int token_count);
    bool PushChunk(
        const float* audio_features,
        int frame_count,
        int active_token,
        float* scores,
        int& scored_token_count,
        float* region_logits = nullptr);
    // Scores the complete transcript lattice for the authoritative fixed-lag
    // Viterbi decoder. This avoids a heuristic moving beam at the cost of a
    // still-small token loop inside the same fused kernel launch.
    bool PushChunkAllTokens(
        const float* audio_features,
        int frame_count,
        float* scores,
        int& scored_token_count,
        float* region_logits = nullptr);
    const std::string& LastError() const;
    RuntimeFootprint Footprint() const;
    RuntimeDeviceInfo DeviceInfo() const;

private:
    struct Impl;
    std::unique_ptr<Impl> State;
};

}  // namespace offgridai::neural_streamer
