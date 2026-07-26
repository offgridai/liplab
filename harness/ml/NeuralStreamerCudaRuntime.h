#pragma once

#include <cstddef>
#include <memory>
#include <string>

namespace offgridai::neural_streamer {

struct RuntimeFootprint {
    std::size_t CheckpointBytes = 0;
    std::size_t WeightBytes = 0;
    std::size_t TokenBytes = 0;
    int StreamPriority = 0;
};

class CudaRuntime {
public:
    CudaRuntime();
    ~CudaRuntime();
    CudaRuntime(const CudaRuntime&) = delete;
    CudaRuntime& operator=(const CudaRuntime&) = delete;

    bool LoadCheckpoint(const std::string& path);
    bool ResetTokens(const float* features, int token_count);
    bool PushChunk(
        const float* audio_features,
        int frame_count,
        int active_token,
        float* scores,
        int& scored_token_count);
    const std::string& LastError() const;
    RuntimeFootprint Footprint() const;

private:
    struct Impl;
    std::unique_ptr<Impl> State;
};

}  // namespace offgridai::neural_streamer
