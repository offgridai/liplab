#include <torch/torch.h>

#include "Lipsync/OffgridAINeuralStreamerRuntime.h"
#include "Lipsync/OffgridAINeuralStreamerModelFormat.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace fs = std::filesystem;

constexpr int kAudioDimensions = offgridai::neural_streamer::kAudioFeatureCount;
constexpr int kPhoneBuckets = offgridai::neural_streamer::kPhoneBuckets;
constexpr int kPoseBuckets = offgridai::neural_streamer::kPoseBuckets;
constexpr int kTokenContinuous = offgridai::neural_streamer::kTokenContinuous;
constexpr int kTokenDimensions = offgridai::neural_streamer::kTokenDimensions;
constexpr int kPunctuationTypeCount =
    offgridai::neural_streamer::kPunctuationTypeCount;
constexpr int kPunctuationFeatureCount =
    offgridai::neural_streamer::kPunctuationFeatureCount;
constexpr int kHiddenDimensions = offgridai::neural_streamer::kHiddenDimensions;
constexpr int kFrameMilliseconds = 10;
constexpr int kStreamingLagFrames = 30;
constexpr int kForwardSumCadence = 16;
constexpr float kInterpolatedPhoneWeight = 0.10f;
constexpr float kWordIntervalLossWeight = 0.35f;

struct TokenRow {
    int EventIndex = -1;
    std::string Pose;
    std::string Phone;
    int SentenceIndex = -1;
    float DurationPrior = 0.075f;
    float IsVowel = 0.0f;
    float VisualRole = 0.0f;
    float TextCenterNorm = 0.0f;
    float Strength = 0.0f;
    float IsSilence = 0.0f;
    float IsWordStart = 0.0f;
    float IsWordEnd = 0.0f;
    float IsRegionStart = 0.0f;
    float IsRegionEnd = 0.0f;
    bool IsSentenceBoundary = false;
    float IsPauseBoundary = 0.0f;
    float HasPunctuation = 0.0f;
    int PunctuationType = 0;
    float TargetCenter = 0.0f;
    float TargetDuration = 0.02f;
    float HasMfaTarget = 0.0f;
};

struct SequenceCase {
    std::string CaseId;
    std::string Split;
    std::vector<float> FrameTimes;
    std::vector<float> Audio;
    std::vector<TokenRow> Tokens;
};

static std::vector<std::string> parse_csv_row(const std::string& line)
{
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char ch = line[i];
        if (ch == '"') {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"') { field.push_back('"'); ++i; }
            else quoted = !quoted;
        } else if (ch == ',' && !quoted) { fields.push_back(field); field.clear(); }
        else field.push_back(ch);
    }
    fields.push_back(field);
    return fields;
}

static std::map<std::string, size_t> columns(const std::string& header)
{
    std::map<std::string, size_t> result;
    const auto fields = parse_csv_row(header);
    for (size_t i = 0; i < fields.size(); ++i) result[fields[i]] = i;
    return result;
}

static const std::string& cell(const std::vector<std::string>& row,
    const std::map<std::string, size_t>& header, const std::string& name)
{
    const auto found = header.find(name);
    if (found == header.end() || found->second >= row.size())
        throw std::runtime_error("missing CSV column: " + name);
    return row[found->second];
}

static std::string json_string_value(const std::string& line)
{
    const size_t colon = line.find(':');
    const size_t first = line.find('"', colon + 1);
    const size_t last = line.find('"', first + 1);
    if (colon == std::string::npos || first == std::string::npos || last == std::string::npos)
        throw std::runtime_error("invalid split manifest line");
    return line.substr(first + 1, last - first - 1);
}

static std::map<std::string, std::string> read_splits(const fs::path& path)
{
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot read split manifest: " + path.string());
    std::map<std::string, std::string> result;
    std::string pending_case;
    std::string line;
    while (std::getline(input, line)) {
        if (line.find("\"case_id\"") != std::string::npos) pending_case = json_string_value(line);
        else if (!pending_case.empty() && line.find("\"split\"") != std::string::npos) {
            result[pending_case] = json_string_value(line);
            pending_case.clear();
        }
    }
    return result;
}

static SequenceCase read_case(const fs::path& case_dir, const std::string& split)
{
    SequenceCase result;
    result.CaseId = case_dir.filename().string();
    result.Split = split;
    {
        std::ifstream input(case_dir / "monotonic_audio_frames.csv");
        if (!input) throw std::runtime_error("missing monotonic frames for " + result.CaseId);
        std::string line;
        std::getline(input, line);
        const auto header = columns(line);
        const std::vector<std::string> feature_names = {
            "log_rms", "rms_norm", "delta_rms", "flux", "zcr", "low_band_norm", "mid_band_norm",
            "high_band_norm", "spectral_centroid_norm", "periodicity", "rich_low_band_norm",
            "rich_mid_band_norm", "rich_high_band_norm", "rich_spectral_centroid_norm",
            "rich_spectral_rolloff_norm", "rich_spectral_flatness", "rich_spectral_flux",
            "rich_periodicity"};
        while (std::getline(input, line)) {
            if (line.empty()) continue;
            const auto row = parse_csv_row(line);
            result.FrameTimes.push_back(0.5f * (std::stof(cell(row, header, "start_sec"))
                + std::stof(cell(row, header, "end_sec"))));
            for (const std::string& name : feature_names)
                result.Audio.push_back(std::stof(cell(row, header, name)));
        }
    }
    {
        std::ifstream input(case_dir / "monotonic_tokens.csv");
        if (!input) throw std::runtime_error("missing monotonic tokens for " + result.CaseId);
        std::string line;
        std::getline(input, line);
        const auto header = columns(line);
        while (std::getline(input, line)) {
            if (line.empty()) continue;
            const auto row = parse_csv_row(line);
            TokenRow token;
            token.EventIndex = std::stoi(cell(row, header, "event_index"));
            token.Pose = cell(row, header, "pose");
            token.Phone = cell(row, header, "phone_base");
            token.SentenceIndex = std::stoi(cell(row, header, "sentence_index"));
            token.DurationPrior = std::stof(cell(row, header, "duration_prior_sec"));
            token.IsVowel = std::stof(cell(row, header, "is_vowel"));
            token.VisualRole = std::stof(cell(row, header, "visual_role"));
            token.TextCenterNorm = std::stof(cell(row, header, "text_center_norm"));
            token.Strength = std::stof(cell(row, header, "strength"));
            token.IsSilence = std::stof(cell(row, header, "is_silence"));
            token.IsWordStart = std::stof(cell(row, header, "is_word_start"));
            token.IsWordEnd = std::stof(cell(row, header, "is_word_end"));
            token.IsRegionStart = std::stof(cell(row, header, "is_region_start"));
            token.IsRegionEnd = std::stof(cell(row, header, "is_region_end"));
            token.IsSentenceBoundary =
                std::stoi(cell(row, header, "is_sentence_boundary")) != 0;
            token.IsPauseBoundary = std::stof(cell(row, header, "is_pause_boundary"));
            token.HasPunctuation = std::stof(cell(row, header, "has_punctuation"));
            token.PunctuationType = std::stoi(cell(row, header, "punctuation_type"));
            token.TargetCenter = std::stof(cell(row, header, "target_center_sec"));
            token.TargetDuration = std::stof(cell(row, header, "target_duration_sec"));
            token.HasMfaTarget = std::stof(cell(row, header, "has_mfa_target"));
            result.Tokens.push_back(std::move(token));
        }
    }
    return result;
}

static std::vector<SequenceCase> read_dataset(const fs::path& root)
{
    const auto splits = read_splits(root / "inputs/timing_split_v1.json");
    const fs::path run_root = root / "outputs/runs/latest";
    std::vector<SequenceCase> result;
    for (const auto& [case_id, split] : splits) {
        const fs::path case_dir = run_root / case_id;
        if (!fs::exists(case_dir / "monotonic_audio_frames.csv")) continue;
        result.push_back(read_case(case_dir, split));
    }
    if (result.size() != splits.size())
        throw std::runtime_error("sequence export does not cover the frozen split");
    return result;
}

static std::uint32_t fnv1a(const std::string& value)
{
    std::uint32_t hash = 2166136261u;
    for (unsigned char ch : value) { hash ^= ch; hash *= 16777619u; }
    return hash;
}

static torch::Tensor token_features(const SequenceCase& sequence)
{
    std::vector<float> values(sequence.Tokens.size() * kTokenDimensions, 0.0f);
    for (size_t i = 0; i < sequence.Tokens.size(); ++i) {
        const TokenRow& token = sequence.Tokens[i];
        float* row = values.data() + i * kTokenDimensions;
        row[fnv1a(token.Phone) % kPhoneBuckets] = 1.0f;
        row[kPhoneBuckets + fnv1a(token.Pose) % kPoseBuckets] = 1.0f;
        float* continuous = row + kPhoneBuckets + kPoseBuckets;
        continuous[0] = token.DurationPrior;
        continuous[1] = token.IsVowel;
        continuous[2] = token.VisualRole / 3.0f;
        continuous[3] = token.TextCenterNorm;
        continuous[4] = token.Strength;
        continuous[5] = i == 0 ? 1.0f : 0.0f;
        continuous[6] = i + 1 == sequence.Tokens.size() ? 1.0f : 0.0f;
        continuous[7] = token.IsSilence;
        continuous[8] = token.IsWordStart;
        continuous[9] = token.IsWordEnd;
        continuous[10] = token.IsPauseBoundary;
        continuous[11] = token.HasPunctuation;
        if (token.PunctuationType > 0
            && token.PunctuationType <= kPunctuationTypeCount) {
            continuous[12 + token.PunctuationType - 1] = 1.0f;
        }
    }
    return torch::from_blob(values.data(), {
        static_cast<std::int64_t>(sequence.Tokens.size()), kTokenDimensions}).clone();
}

struct CaseTensors {
    torch::Tensor Audio;
    torch::Tensor Tokens;
    torch::Tensor Target;
    torch::Tensor FrameWeights;
    torch::Tensor CurriculumFrameWeights;
    torch::Tensor RegionTarget;
    torch::Tensor RegionWeights;
    torch::Tensor FrameTimes;
    int BeginFrame = 0;
    int EndFrame = 0;
};

static CaseTensors tensors(const SequenceCase& sequence)
{
    const int frame_count = static_cast<int>(sequence.FrameTimes.size());
    const int token_count = static_cast<int>(sequence.Tokens.size());
    CaseTensors result;
    // Keep the complete audio stream. Cropping from detector occupancy made
    // the nominally neural path depend on the detector before inference began.
    result.BeginFrame = 0;
    result.EndFrame = frame_count;
    const int used_frames = result.EndFrame - result.BeginFrame;
    result.Audio = torch::from_blob(
        const_cast<float*>(sequence.Audio.data()) + result.BeginFrame * kAudioDimensions,
        {used_frames, kAudioDimensions}).clone().transpose(0, 1).unsqueeze(0);
    result.FrameTimes = torch::from_blob(
        const_cast<float*>(sequence.FrameTimes.data()) + result.BeginFrame, {used_frames}).clone();
    result.Tokens = token_features(sequence);
    std::vector<std::int64_t> targets;
    targets.reserve(used_frames);
    int token = 0;
    for (int frame = result.BeginFrame; frame < result.EndFrame; ++frame) {
        const float time = sequence.FrameTimes[frame];
        while (token + 1 < token_count) {
            const TokenRow& current = sequence.Tokens[token];
            const TokenRow& next = sequence.Tokens[token + 1];
            float boundary = 0.5f * (current.TargetCenter + next.TargetCenter);
            if (next.IsWordStart > 0.5f)
                boundary = next.TargetCenter - 0.5f * next.TargetDuration;
            else if (next.IsSilence > 0.5f)
                boundary = current.TargetCenter + 0.5f * current.TargetDuration;
            if (time <= boundary) break;
            ++token;
        }
        targets.push_back(token);
    }
    result.Target = torch::from_blob(targets.data(), {used_frames},
        torch::TensorOptions().dtype(torch::kInt64)).clone();
    std::vector<float> frame_weights;
    std::vector<float> curriculum_frame_weights;
    std::vector<float> region_targets;
    std::vector<float> region_weights;
    frame_weights.reserve(used_frames);
    curriculum_frame_weights.reserve(used_frames);
    for (std::int64_t target : targets) {
        const TokenRow& selected = sequence.Tokens[static_cast<size_t>(target)];
        float weight = 1.0f;
        float curriculum_weight = 1.0f;
        if (selected.IsVowel > 0.5f) {
            weight = std::max(weight, 2.0f);
            curriculum_weight = std::max(curriculum_weight, 3.0f);
        }
        if (selected.IsSilence > 0.5f) weight = 5.0f;
        if (selected.IsSilence > 0.5f) curriculum_weight = 8.0f;
        if (selected.IsWordStart > 0.5f) weight = std::max(weight, 4.0f);
        if (selected.IsWordStart > 0.5f) curriculum_weight = std::max(curriculum_weight, 6.0f);
        if (selected.IsWordEnd > 0.5f) weight = std::max(weight, 4.0f);
        if (selected.IsWordEnd > 0.5f) curriculum_weight = std::max(curriculum_weight, 6.0f);
        if (selected.IsRegionStart > 0.5f || selected.IsRegionEnd > 0.5f)
            weight = std::max(weight, 7.0f);
        if (selected.IsRegionStart > 0.5f || selected.IsRegionEnd > 0.5f)
            curriculum_weight = std::max(curriculum_weight, 12.0f);
        if (selected.IsSentenceBoundary) {
            weight = std::max(weight, 8.0f);
            curriculum_weight = std::max(curriculum_weight, 12.0f);
        }
        frame_weights.push_back(weight);
        curriculum_frame_weights.push_back(curriculum_weight);
        const bool outside_region = selected.IsSilence > 0.5f
            && (selected.IsRegionStart > 0.5f || selected.IsRegionEnd > 0.5f);
        region_targets.push_back(outside_region ? 0.0f : 1.0f);
        region_weights.push_back(outside_region ? 2.0f : 1.0f);
    }
    // Occupancy weighting alone spends most of its gradient on token interiors.
    // Put a narrow, symmetric band around the transitions that drive the
    // product metrics. Region flags are MFA supervision only; they are never
    // included in token_features() or the runtime checkpoint inputs.
    for (int frame = 1; frame < used_frames; ++frame) {
        const int left = static_cast<int>(targets[static_cast<size_t>(frame - 1)]);
        const int right = static_cast<int>(targets[static_cast<size_t>(frame)]);
        if (left == right) continue;
        const TokenRow& before = sequence.Tokens[static_cast<size_t>(left)];
        const TokenRow& after = sequence.Tokens[static_cast<size_t>(right)];
        float boundary_weight = 1.0f;
        float curriculum_boundary_weight = 1.0f;
        if (before.IsVowel > 0.5f || after.IsVowel > 0.5f) {
            boundary_weight = std::max(boundary_weight, 4.0f);
            curriculum_boundary_weight = std::max(curriculum_boundary_weight, 6.0f);
        }
        if (after.IsWordStart > 0.5f) {
            boundary_weight = std::max(boundary_weight, 8.0f);
            curriculum_boundary_weight = std::max(curriculum_boundary_weight, 10.0f);
        }
        if (before.IsWordEnd > 0.5f) {
            boundary_weight = std::max(boundary_weight, 8.0f);
            curriculum_boundary_weight = std::max(curriculum_boundary_weight, 10.0f);
        }
        if (before.IsSilence > 0.5f || after.IsSilence > 0.5f) {
            boundary_weight = std::max(boundary_weight, 10.0f);
            curriculum_boundary_weight = std::max(curriculum_boundary_weight, 14.0f);
        }
        if (before.IsRegionEnd > 0.5f || after.IsRegionStart > 0.5f) {
            boundary_weight = std::max(boundary_weight, 14.0f);
            curriculum_boundary_weight = std::max(curriculum_boundary_weight, 18.0f);
        }
        if (before.IsSentenceBoundary || after.IsSentenceBoundary) {
            boundary_weight = std::max(boundary_weight, 14.0f);
            curriculum_boundary_weight = std::max(
                curriculum_boundary_weight, 18.0f);
        }
        constexpr int kBoundaryRadiusFrames = 4;
        for (int nearby = std::max(0, frame - kBoundaryRadiusFrames);
             nearby < std::min(used_frames, frame + kBoundaryRadiusFrames + 1); ++nearby) {
            frame_weights[static_cast<size_t>(nearby)] = std::max(
                frame_weights[static_cast<size_t>(nearby)], boundary_weight);
            curriculum_frame_weights[static_cast<size_t>(nearby)] = std::max(
                curriculum_frame_weights[static_cast<size_t>(nearby)],
                curriculum_boundary_weight);
        }
    }
    // A planned token can lack a matching MFA phone when the synthesizer uses
    // a different pronunciation (for example, reduced vowels). Its target
    // center is then interpolated so the monotonic path remains complete, but
    // that interpolation is not evidence that the planned phone was present
    // in the audio. Keep a small path-shaping weight while preventing those
    // fabricated labels from dominating acoustic phone identity learning.
    for (int frame = 0; frame < used_frames; ++frame) {
        const TokenRow& selected = sequence.Tokens[
            static_cast<size_t>(targets[static_cast<size_t>(frame)])];
        if (selected.HasMfaTarget > 0.5f || selected.IsSilence > 0.5f) continue;
        frame_weights[static_cast<size_t>(frame)] *= kInterpolatedPhoneWeight;
        curriculum_frame_weights[static_cast<size_t>(frame)] *= kInterpolatedPhoneWeight;
    }
    result.FrameWeights = torch::from_blob(frame_weights.data(), {used_frames}).clone();
    result.CurriculumFrameWeights = torch::from_blob(
        curriculum_frame_weights.data(), {used_frames}).clone();
    for (int frame = 1; frame < used_frames; ++frame) {
        if (region_targets[static_cast<size_t>(frame)]
            == region_targets[static_cast<size_t>(frame - 1)]) continue;
        constexpr int kRegionBoundaryRadiusFrames = 10;
        for (int nearby = std::max(0, frame - kRegionBoundaryRadiusFrames);
             nearby < std::min(used_frames, frame + kRegionBoundaryRadiusFrames + 1); ++nearby)
            region_weights[static_cast<size_t>(nearby)] = std::max(
                region_weights[static_cast<size_t>(nearby)], 6.0f);
    }
    result.RegionTarget = torch::from_blob(region_targets.data(), {used_frames}).clone();
    result.RegionWeights = torch::from_blob(region_weights.data(), {used_frames}).clone();
    return result;
}

struct MonotonicNetworkImpl : torch::nn::Module {
    MonotonicNetworkImpl()
    {
        AudioConv1 = register_module("audio_conv1", torch::nn::Conv1d(
            torch::nn::Conv1dOptions(kAudioDimensions, kHiddenDimensions, 3)
                .padding(2 * offgridai::neural_streamer::kAudioConv1Dilation)
                .dilation(offgridai::neural_streamer::kAudioConv1Dilation)));
        AudioConv2 = register_module("audio_conv2", torch::nn::Conv1d(
            torch::nn::Conv1dOptions(kHiddenDimensions, kHiddenDimensions, 3)
                .padding(2 * offgridai::neural_streamer::kAudioConv2Dilation)
                .dilation(offgridai::neural_streamer::kAudioConv2Dilation)));
        RegionConv1 = register_module("region_conv1", torch::nn::Conv1d(
            torch::nn::Conv1dOptions(kAudioDimensions, kHiddenDimensions, 3)
                .padding(2 * offgridai::neural_streamer::kRegionConv1Dilation)
                .dilation(offgridai::neural_streamer::kRegionConv1Dilation)));
        RegionConv2 = register_module("region_conv2", torch::nn::Conv1d(
            torch::nn::Conv1dOptions(kHiddenDimensions, kHiddenDimensions, 3)
                .padding(2 * offgridai::neural_streamer::kRegionConv2Dilation)
                .dilation(offgridai::neural_streamer::kRegionConv2Dilation)));
        TokenLinear1 = register_module("token_linear_1",
            torch::nn::Linear(kTokenDimensions, kHiddenDimensions));
        TokenLinear2 = register_module("token_linear_2",
            torch::nn::Linear(kHiddenDimensions, kHiddenDimensions));
        RegionLinear = register_module("region_linear",
            torch::nn::Linear(kHiddenDimensions, 1));
        AudioMean = register_buffer("audio_mean", torch::zeros({1, kAudioDimensions, 1}));
        AudioScale = register_buffer("audio_scale", torch::ones({1, kAudioDimensions, 1}));
    }

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor audio, torch::Tensor token)
    {
        const std::int64_t frames = audio.size(2);
        audio = (audio - AudioMean) / AudioScale;
        if (is_training()) audio = audio + 0.015f * torch::randn_like(audio);
        torch::Tensor audio_state = torch::relu(AudioConv1->forward(audio));
        audio_state = audio_state.slice(2, 0, frames);
        audio_state = torch::relu(AudioConv2->forward(audio_state));
        audio_state = audio_state.slice(2, 0, frames).squeeze(0).transpose(0, 1);
        torch::Tensor region_state = torch::relu(RegionConv1->forward(audio));
        region_state = region_state.slice(2, 0, frames);
        region_state = torch::relu(RegionConv2->forward(region_state));
        region_state = region_state.slice(2, 0, frames).squeeze(0).transpose(0, 1);
        torch::Tensor token_state = TokenLinear2->forward(torch::relu(TokenLinear1->forward(token)));
        torch::Tensor region_logits = RegionLinear->forward(region_state).squeeze(1);
        const torch::Tensor silence_mask = token.index({torch::indexing::Slice(),
            kPhoneBuckets + kPoseBuckets + 7}).unsqueeze(0);
        const torch::Tensor occupancy_bias = 2.0f * torch::tanh(region_logits.detach()).unsqueeze(1)
            * (1.0f - 2.0f * silence_mask);
        return {
            torch::matmul(audio_state, token_state.transpose(0, 1)) / 8.0f + occupancy_bias,
            region_logits
        };
    }

    torch::nn::Conv1d AudioConv1{nullptr};
    torch::nn::Conv1d AudioConv2{nullptr};
    torch::nn::Conv1d RegionConv1{nullptr};
    torch::nn::Conv1d RegionConv2{nullptr};
    torch::nn::Linear TokenLinear1{nullptr};
    torch::nn::Linear TokenLinear2{nullptr};
    torch::nn::Linear RegionLinear{nullptr};
    torch::Tensor AudioMean;
    torch::Tensor AudioScale;
};
TORCH_MODULE(MonotonicNetwork);

static std::vector<int> viterbi_path(
    const torch::Tensor& score_tensor,
    const torch::Tensor& region_logit_tensor,
    const std::vector<TokenRow>& tokens,
    int fixed_lag_frames);

static void append_tensor(std::vector<float>& values, const torch::Tensor& tensor)
{
    const torch::Tensor contiguous = tensor.detach().to(torch::kCPU, torch::kFloat32).contiguous();
    const float* begin = contiguous.data_ptr<float>();
    values.insert(values.end(), begin, begin + contiguous.numel());
}

static void write_runtime_checkpoint(const fs::path& path, const MonotonicNetwork& model)
{
    using namespace offgridai::neural_streamer;
    std::vector<float> values;
    values.reserve(kCheckpointFloatCount);
    append_tensor(values, model->AudioMean);
    append_tensor(values, model->AudioScale);
    append_tensor(values, model->AudioConv1->weight);
    append_tensor(values, model->AudioConv1->bias);
    append_tensor(values, model->AudioConv2->weight);
    append_tensor(values, model->AudioConv2->bias);
    append_tensor(values, model->TokenLinear1->weight);
    append_tensor(values, model->TokenLinear1->bias);
    append_tensor(values, model->TokenLinear2->weight);
    append_tensor(values, model->TokenLinear2->bias);
    append_tensor(values, model->RegionConv1->weight);
    append_tensor(values, model->RegionConv1->bias);
    append_tensor(values, model->RegionConv2->weight);
    append_tensor(values, model->RegionConv2->bias);
    append_tensor(values, model->RegionLinear->weight);
    append_tensor(values, model->RegionLinear->bias);
    if (values.size() != kCheckpointFloatCount)
        throw std::runtime_error("runtime checkpoint parameter count changed");
    const CheckpointHeader header;
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("cannot write runtime checkpoint");
    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    output.write(reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!output) throw std::runtime_error("runtime checkpoint write failed");
}

static void load_runtime_checkpoint(const fs::path& path, MonotonicNetwork& model)
{
    using namespace offgridai::neural_streamer;
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read initial runtime checkpoint: " + path.string());
    CheckpointHeader header;
    input.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!input || header.Magic != kCheckpointMagic || header.Version != kCheckpointVersion
        || header.AudioFeatures != kAudioFeatureCount
        || header.TokenFeatures != offgridai::neural_streamer::kTokenDimensions
        || header.HiddenDimensions != offgridai::neural_streamer::kHiddenDimensions
        || header.ParameterCount != kCheckpointFloatCount)
        throw std::runtime_error("initial runtime checkpoint schema mismatch");
    std::vector<float> values(kCheckpointFloatCount);
    input.read(reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!input) throw std::runtime_error("initial runtime checkpoint is truncated");
    std::size_t offset = 0;
    torch::NoGradGuard no_grad;
    auto restore = [&](torch::Tensor tensor) {
        const std::size_t count = static_cast<std::size_t>(tensor.numel());
        const torch::Tensor source = torch::from_blob(
            values.data() + offset, tensor.sizes(), torch::TensorOptions().dtype(torch::kFloat32));
        tensor.copy_(source);
        offset += count;
    };
    restore(model->AudioMean);
    restore(model->AudioScale);
    restore(model->AudioConv1->weight);
    restore(model->AudioConv1->bias);
    restore(model->AudioConv2->weight);
    restore(model->AudioConv2->bias);
    restore(model->TokenLinear1->weight);
    restore(model->TokenLinear1->bias);
    restore(model->TokenLinear2->weight);
    restore(model->TokenLinear2->bias);
    restore(model->RegionConv1->weight);
    restore(model->RegionConv1->bias);
    restore(model->RegionConv2->weight);
    restore(model->RegionConv2->bias);
    restore(model->RegionLinear->weight);
    restore(model->RegionLinear->bias);
    if (offset != values.size())
        throw std::runtime_error("initial runtime checkpoint parameter count changed");
}

static double percentile(std::vector<double> values, double fraction)
{
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double position = std::clamp(fraction, 0.0, 1.0) * (values.size() - 1);
    const std::size_t low = static_cast<std::size_t>(std::floor(position));
    const std::size_t high = static_cast<std::size_t>(std::ceil(position));
    const double weight = position - low;
    return values[low] * (1.0 - weight) + values[high] * weight;
}

struct RuntimeBenchmark {
    double MeanMicroseconds = 0.0;
    double MedianMicroseconds = 0.0;
    double P95Microseconds = 0.0;
    double MaxMicroseconds = 0.0;
    double TokenSetupMicroseconds = 0.0;
    double MaxLogitError = 0.0;
    int Calls = 0;
    offgridai::neural_streamer::RuntimeFootprint Footprint;
};

static RuntimeBenchmark benchmark_runtime(
    const fs::path& checkpoint,
    MonotonicNetwork& model,
    const std::vector<SequenceCase>& dataset,
    const torch::Device& device)
{
    using clock = std::chrono::steady_clock;
    using namespace offgridai::neural_streamer;
    const auto selected = std::find_if(dataset.begin(), dataset.end(), [](const SequenceCase& sequence) {
        return sequence.Split == "validation"
            && sequence.FrameTimes.size() >= 32
            && sequence.Tokens.size() >= kForwardTokenWindow;
    });
    if (selected == dataset.end()) throw std::runtime_error("no runtime benchmark sequence");
    CaseTensors row = tensors(*selected);
    model->eval();
    torch::InferenceMode inference_guard;
    const auto reference_output = model->forward(
        row.Audio.to(device), row.Tokens.to(device));
    const torch::Tensor reference = reference_output.first.to(torch::kCPU).contiguous();
    const torch::Tensor reference_regions =
        reference_output.second.to(torch::kCPU).contiguous();
    const std::vector<int> path = viterbi_path(
        reference, reference_regions, selected->Tokens, kStreamingLagFrames);
    const torch::Tensor frame_major = row.Audio.squeeze(0).transpose(0, 1).contiguous();
    const float* frames = frame_major.data_ptr<float>();
    const int frame_count = static_cast<int>(frame_major.size(0));
    const int token_count = static_cast<int>(row.Tokens.size(0));

    CudaRuntime runtime;
    if (!runtime.LoadCheckpoint(checkpoint.string()))
        throw std::runtime_error(runtime.LastError());
    const auto token_begin = clock::now();
    if (!runtime.ResetTokens(row.Tokens.contiguous().data_ptr<float>(), token_count))
        throw std::runtime_error(runtime.LastError());
    const auto token_end = clock::now();

    RuntimeBenchmark result;
    result.TokenSetupMicroseconds = std::chrono::duration<double, std::micro>(
        token_end - token_begin).count();
    const auto reference_values = reference.accessor<float, 2>();
    std::array<float, kRuntimeChunkFrames * kForwardTokenWindow> scores{};
    std::array<float, kRuntimeChunkFrames> region_logits{};
    for (int begin = 0; begin < frame_count; begin += kRuntimeChunkFrames) {
        const int count = std::min(kRuntimeChunkFrames, frame_count - begin);
        const int active = std::clamp(path[static_cast<std::size_t>(begin)],
            0, token_count - 1);
        int scored_tokens = 0;
        if (!runtime.PushChunk(frames + begin * kAudioFeatureCount, count,
                active, scores.data(), scored_tokens, region_logits.data()))
            throw std::runtime_error(runtime.LastError());
        for (int frame = 0; frame < count; ++frame)
            for (int token = 0; token < scored_tokens; ++token)
                result.MaxLogitError = std::max(result.MaxLogitError,
                    std::abs(static_cast<double>(scores[frame * kForwardTokenWindow + token])
                        - reference_values[begin + frame][active + token]));
        const auto reference_region_values = reference_regions.accessor<float, 1>();
        for (int frame = 0; frame < count; ++frame)
            result.MaxLogitError = std::max(result.MaxLogitError,
                std::abs(static_cast<double>(region_logits[frame])
                    - reference_region_values[begin + frame]));
    }
    if (result.MaxLogitError > 0.10)
        throw std::runtime_error("FP16 CUDA runtime diverges from LibTorch reference; max error "
            + std::to_string(result.MaxLogitError));

    std::vector<double> samples;
    samples.reserve(512);
    int begin = 0;
    for (int call = 0; call < 528; ++call) {
        if (begin >= frame_count) {
            begin = 0;
            if (!runtime.ResetTokens(row.Tokens.contiguous().data_ptr<float>(), token_count))
                throw std::runtime_error(runtime.LastError());
        }
        const int count = std::min(kRuntimeChunkFrames, frame_count - begin);
        const int active = std::clamp(path[static_cast<std::size_t>(begin)],
            0, token_count - 1);
        int scored_tokens = 0;
        const auto start = clock::now();
        if (!runtime.PushChunk(frames + begin * kAudioFeatureCount, count,
                active, scores.data(), scored_tokens, region_logits.data()))
            throw std::runtime_error(runtime.LastError());
        const auto end = clock::now();
        if (call >= 16)
            samples.push_back(std::chrono::duration<double, std::micro>(end - start).count());
        begin += count;
    }
    result.Calls = static_cast<int>(samples.size());
    result.MeanMicroseconds = std::accumulate(samples.begin(), samples.end(), 0.0)
        / std::max<std::size_t>(samples.size(), 1);
    result.MedianMicroseconds = percentile(samples, 0.50);
    result.P95Microseconds = percentile(samples, 0.95);
    result.MaxMicroseconds = *std::max_element(samples.begin(), samples.end());
    result.Footprint = runtime.Footprint();
    return result;
}

static std::vector<int> viterbi_path(const torch::Tensor& score_tensor,
    const torch::Tensor& region_logit_tensor,
    const std::vector<TokenRow>& tokens, int fixed_lag_frames)
{
    const torch::Tensor scores = torch::log_softmax(score_tensor, 1).to(torch::kCPU);
    const torch::Tensor speech_probability = torch::sigmoid(region_logit_tensor)
        .to(torch::kCPU).clamp(0.02f, 0.98f);
    const int frames = static_cast<int>(scores.size(0));
    const int count = static_cast<int>(scores.size(1));
    const auto access = scores.accessor<float, 2>();
    const auto occupancy = speech_probability.accessor<float, 1>();
    const float negative = -1.0e30f;
    std::vector<float> previous(count, negative), current(count, negative);
    std::vector<int> previous_dwell(count, 0), current_dwell(count, 0);
    std::vector<std::uint8_t> advanced(static_cast<size_t>(frames) * count, 0);
    constexpr float kOccupancyLogWeight = 1.75f;
    previous[0] = access[0][0]
        + kOccupancyLogWeight * std::log(std::max(1.0f - occupancy[0], 0.02f));
    previous_dwell[0] = 1;
    std::vector<int> path(frames, -1);
    path[0] = 0;
    for (int frame = 1; frame < frames; ++frame) {
        std::fill(current.begin(), current.end(), negative);
        std::fill(current_dwell.begin(), current_dwell.end(), 0);
        for (int token = 0; token < count; ++token) {
            const float occupancy_probability = tokens[token].IsSilence > 0.5f
                ? 1.0f - occupancy[frame]
                : occupancy[frame];
            const float emission = access[frame][token]
                + kOccupancyLogWeight * std::log(std::clamp(
                    occupancy_probability, 0.02f, 0.98f));
            const auto minimum_dwell = [&](int candidate) {
                if (tokens[candidate].IsSentenceBoundary) return 8;
                if (tokens[candidate].IsSilence > 0.5f) return 1;
                return std::clamp(
                    static_cast<int>(std::ceil(tokens[candidate].DurationPrior * 50.0f)),
                    2, 8);
            };
            const float advance_probability = std::clamp(
                0.010f / std::max(tokens[token].DurationPrior, 0.020f), 0.02f, 0.80f);
            float stay = previous[token] + std::log1p(-advance_probability);
            const int maximum_dwell = std::clamp(
                static_cast<int>(std::ceil(tokens[token].DurationPrior * 300.0f)),
                6, 30);
            if (tokens[token].IsSilence <= 0.5f
                && previous_dwell[token] >= maximum_dwell) {
                stay = negative;
            }
            float advance = negative;
            if (token > 0) {
                const int missing_dwell = std::max(
                    minimum_dwell(token - 1) - previous_dwell[token - 1], 0);
                const float early_penalty = missing_dwell
                    * (tokens[token - 1].IsSentenceBoundary ? 0.75f : 0.50f);
                if (tokens[token - 1].IsSilence > 0.5f
                    || previous_dwell[token - 1] >= 2) {
                    advance = previous[token - 1] + std::log(std::clamp(
                        0.010f / std::max(tokens[token - 1].DurationPrior, 0.020f),
                        0.02f, 0.80f)) - early_penalty;
                }
            }
            if (advance > stay) {
                current[token] = advance + emission;
                current_dwell[token] = 1;
                advanced[static_cast<size_t>(frame) * count + token] = 1;
            } else {
                current[token] = stay + emission;
                current_dwell[token] = previous_dwell[token] + 1;
            }
        }
        previous.swap(current);
        previous_dwell.swap(current_dwell);

        // Commit only the state far enough behind the live frontier. This is
        // the fixed-lag streaming approximation: it never consults frames
        // later than commit_frame + fixed_lag_frames.
        if (fixed_lag_frames >= 0 && frame >= fixed_lag_frames) {
            int state = static_cast<int>(std::distance(previous.begin(),
                std::max_element(previous.begin(), previous.end())));
            for (int back = frame; back > frame - fixed_lag_frames; --back) {
                if (state > 0 && advanced[static_cast<size_t>(back) * count + state]) --state;
            }
            const int commit_frame = frame - fixed_lag_frames;
            if (commit_frame > 0)
                state = std::clamp(state, path[commit_frame - 1], path[commit_frame - 1] + 1);
            path[commit_frame] = state;
        }
    }

    // Flush only the uncommitted lag window when the utterance closes. The
    // final-state requirement guarantees that every transcript token appears.
    int token = count - 1;
    std::vector<int> tail(frames, 0);
    for (int frame = frames - 1; frame >= 0; --frame) {
        tail[frame] = token;
        if (frame > 0 && token > 0 && advanced[static_cast<size_t>(frame) * count + token]) --token;
    }
    if (fixed_lag_frames < 0) return tail;
    const int tail_begin = std::max(0, frames - fixed_lag_frames);
    for (int frame = tail_begin; frame < frames; ++frame) {
        int state = tail[frame];
        if (frame > 0) state = std::clamp(state, path[frame - 1], path[frame - 1] + 1);
        path[frame] = state;
    }
    // A short or low-confidence tail may not connect to the forced endpoint.
    // Preserve completeness without target-time leakage by distributing only
    // the still-required one-step advances across the closing lag window.
    if (path.back() < count - 1) {
        const int prefix_frame = tail_begin - 1;
        const int prefix_state = prefix_frame >= 0 ? path[prefix_frame] : 0;
        const int tail_frames = frames - tail_begin;
        const int remaining = count - 1 - prefix_state;
        if (remaining > tail_frames)
            throw std::runtime_error("streaming lag is too short to close monotonic path");
        for (int offset = 0; offset < tail_frames; ++offset) {
            const int completed = std::min(remaining,
                static_cast<int>(std::floor((offset + 1.0) * remaining / tail_frames)));
            path[tail_begin + offset] = prefix_state + completed;
        }
    }
    return path;
}

static torch::Tensor forward_sum_loss(const torch::Tensor& score_tensor,
    const std::vector<TokenRow>& tokens)
{
    const torch::Tensor log_probability = torch::log_softmax(score_tensor, 1);
    const int frames = static_cast<int>(log_probability.size(0));
    const int count = static_cast<int>(log_probability.size(1));
    std::vector<float> stay_values(count), advance_values(count);
    for (int token = 0; token < count; ++token) {
        const float advance_probability = std::clamp(
            0.010f / std::max(tokens[token].DurationPrior, 0.020f), 0.02f, 0.80f);
        stay_values[token] = std::log1p(-advance_probability);
        advance_values[token] = std::log(advance_probability);
    }
    const torch::Tensor stay_prior = torch::from_blob(stay_values.data(), {count}).clone()
        .to(log_probability.device());
    const torch::Tensor advance_prior = torch::from_blob(advance_values.data(), {count}).clone()
        .to(log_probability.device());
    torch::Tensor alpha = torch::cat({
        log_probability.index({0, 0}).reshape({1}),
        torch::full({count - 1}, -1.0e30f, log_probability.options())});
    for (int frame = 1; frame < frames; ++frame) {
        const torch::Tensor stay = alpha + stay_prior;
        const torch::Tensor advance = torch::cat({
            torch::full({1}, -1.0e30f, log_probability.options()),
            alpha.slice(0, 0, count - 1) + advance_prior.slice(0, 0, count - 1)});
        alpha = torch::logaddexp(stay, advance) + log_probability.index({frame});
    }
    return -alpha.index({count - 1}) / static_cast<float>(frames);
}

static torch::Tensor word_interval_loss(
    const torch::Tensor& score_tensor,
    const torch::Tensor& target_tensor,
    const std::vector<TokenRow>& tokens)
{
    const torch::Tensor probability = torch::softmax(score_tensor, 1);
    std::vector<torch::Tensor> losses;
    for (int first = 0; first < static_cast<int>(tokens.size()); ++first) {
        if (tokens[static_cast<size_t>(first)].IsWordStart <= 0.5f) continue;
        int last = first;
        while (last + 1 < static_cast<int>(tokens.size())
            && tokens[static_cast<size_t>(last)].IsWordEnd <= 0.5f)
            ++last;
        const torch::Tensor word_probability = probability.slice(1, first, last + 1)
            .sum(1).clamp(1.0e-6f, 1.0f - 1.0e-6f);
        const torch::Tensor positive_mask = (target_tensor >= first)
            & (target_tensor <= last);
        const torch::Tensor positive_weights = positive_mask.to(torch::kFloat32);
        torch::Tensor loss = (-torch::log(word_probability) * positive_weights).sum()
            / positive_weights.sum().clamp_min(1.0f);

        // The silence tokens immediately surrounding a word are the direct
        // acoustic alternatives at its entry and exit. Contrast against only
        // those local states so long utterances do not swamp a short word with
        // irrelevant distant negatives.
        torch::Tensor negative_mask = torch::zeros_like(positive_mask);
        if (first > 0)
            negative_mask = negative_mask | (target_tensor == first - 1);
        if (last + 1 < static_cast<int>(tokens.size()))
            negative_mask = negative_mask | (target_tensor == last + 1);
        const torch::Tensor negative_weights = negative_mask.to(torch::kFloat32);
        loss = loss + 0.5f
            * (-torch::log(1.0f - word_probability) * negative_weights).sum()
            / negative_weights.sum().clamp_min(1.0f);
        losses.push_back(loss);
        first = last;
    }
    if (losses.empty()) return torch::zeros({}, score_tensor.options());
    return torch::stack(losses).mean();
}

struct DecodedValidationMetrics {
    int Words = 0;
    int ZeroOverlapWords = 0;
    double MeanOnsetAbsErrorMs = 0.0;
    double MeanExitAbsErrorMs = 0.0;
    double MeanCoverage = 0.0;
    double SelectionScore = 0.0;
};

static DecodedValidationMetrics validation_decoded_metrics(
    MonotonicNetwork& model,
    const std::vector<SequenceCase>& dataset,
    const torch::Device& device)
{
    torch::NoGradGuard no_grad;
    model->eval();
    DecodedValidationMetrics result;
    double onset_total = 0.0;
    double exit_total = 0.0;
    double coverage_total = 0.0;
    for (const SequenceCase& sequence : dataset) {
        if (sequence.Split != "validation") continue;
        CaseTensors row = tensors(sequence);
        const auto output = model->forward(
            row.Audio.to(device), row.Tokens.to(device));
        const std::vector<int> path = viterbi_path(
            output.first, output.second, sequence.Tokens, kStreamingLagFrames);
        const torch::Tensor target_cpu = row.Target.to(torch::kCPU).contiguous();
        const auto target = target_cpu.accessor<std::int64_t, 1>();
        for (int first = 0; first < static_cast<int>(sequence.Tokens.size()); ++first) {
            if (sequence.Tokens[static_cast<size_t>(first)].IsWordStart <= 0.5f)
                continue;
            int last = first;
            while (last + 1 < static_cast<int>(sequence.Tokens.size())
                && sequence.Tokens[static_cast<size_t>(last)].IsWordEnd <= 0.5f)
                ++last;
            int target_first = -1;
            int target_last = -1;
            int predicted_first = -1;
            int predicted_last = -1;
            for (int frame = 0; frame < static_cast<int>(path.size()); ++frame) {
                const bool target_in_word = target[frame] >= first && target[frame] <= last;
                const bool predicted_in_word = path[static_cast<size_t>(frame)] >= first
                    && path[static_cast<size_t>(frame)] <= last;
                if (target_in_word) {
                    if (target_first < 0) target_first = frame;
                    target_last = frame;
                }
                if (predicted_in_word) {
                    if (predicted_first < 0) predicted_first = frame;
                    predicted_last = frame;
                }
            }
            if (target_first < 0 || predicted_first < 0) continue;
            const int overlap = std::max(0,
                std::min(target_last, predicted_last)
                    - std::max(target_first, predicted_first) + 1);
            const int target_frames = target_last - target_first + 1;
            onset_total += std::abs(predicted_first - target_first)
                * kFrameMilliseconds;
            exit_total += std::abs(predicted_last - target_last)
                * kFrameMilliseconds;
            coverage_total += static_cast<double>(overlap) / target_frames;
            result.ZeroOverlapWords += overlap == 0;
            ++result.Words;
            first = last;
        }
    }
    if (result.Words > 0) {
        result.MeanOnsetAbsErrorMs = onset_total / result.Words;
        result.MeanExitAbsErrorMs = exit_total / result.Words;
        result.MeanCoverage = coverage_total / result.Words;
        const double zero_rate = static_cast<double>(result.ZeroOverlapWords)
            / result.Words;
        result.SelectionScore = result.MeanOnsetAbsErrorMs
            + result.MeanExitAbsErrorMs
            + 200.0 * (1.0 - result.MeanCoverage)
            + 400.0 * zero_rate;
    }
    return result;
}

static double validation_loss(MonotonicNetwork& model, const std::vector<SequenceCase>& dataset,
    const torch::Device& device)
{
    torch::NoGradGuard no_grad;
    model->eval();
    double total = 0.0;
    int count = 0;
    for (const SequenceCase& sequence : dataset) {
        if (sequence.Split != "validation") continue;
        CaseTensors row = tensors(sequence);
        const auto output = model->forward(row.Audio.to(device), row.Tokens.to(device));
        const torch::Tensor scores = output.first;
        const torch::Tensor per_frame = torch::nn::functional::cross_entropy(
            scores, row.Target.to(device),
            torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kNone));
        const torch::Tensor weights = row.FrameWeights.to(device);
        const torch::Tensor region_weights = row.RegionWeights.to(device);
        const torch::Tensor region_loss = torch::nn::functional::binary_cross_entropy_with_logits(
            output.second, row.RegionTarget.to(device),
            torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions()
                .weight(region_weights).reduction(torch::kSum)) / region_weights.sum();
        const torch::Tensor interval_loss = word_interval_loss(
            scores, row.Target.to(device), sequence.Tokens);
        total += (((per_frame * weights).sum() / weights.sum())
            + 0.45f * region_loss
            + kWordIntervalLossWeight * interval_loss).item<double>();
        ++count;
    }
    return total / std::max(count, 1);
}

int main(int argc, char** argv)
{
    try {
        const fs::path root = argc > 1 ? fs::absolute(argv[1]) : fs::current_path();
        const fs::path initial_checkpoint = argc > 2 ? fs::absolute(argv[2]) : fs::path();
        const bool fine_tuning = !initial_checkpoint.empty();
        const bool full_fine_tuning = fine_tuning && argc > 3
            && std::string(argv[3]) == "--full-finetune";
        const bool acoustic_fine_tuning = fine_tuning && argc > 3
            && std::string(argv[3]) == "--acoustic-finetune";
        const bool identity_fine_tuning = fine_tuning && argc > 3
            && std::string(argv[3]) == "--identity-finetune";
        const bool word_interval_fine_tuning = fine_tuning && argc > 3
            && std::string(argv[3]) == "--word-interval-finetune";
        std::vector<SequenceCase> dataset = read_dataset(root);
#ifdef _WIN32
        _putenv_s("CUBLAS_WORKSPACE_CONFIG", ":4096:8");
#endif
        torch::manual_seed(0x4c49504c);
        at::globalContext().setDeterministicAlgorithms(true, false);
        const torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        std::cout << "Training neural streamer on " << device
            << " with " << dataset.size() << " utterances\n";
        MonotonicNetwork model;
        if (fine_tuning) {
            load_runtime_checkpoint(initial_checkpoint, model);
            if (acoustic_fine_tuning || identity_fine_tuning
                || word_interval_fine_tuning) {
                for (torch::Tensor parameter : model->parameters())
                    parameter.set_requires_grad(false);
                if (acoustic_fine_tuning) {
                    for (torch::Tensor parameter : model->AudioConv1->parameters())
                        parameter.set_requires_grad(true);
                    for (torch::Tensor parameter : model->AudioConv2->parameters())
                        parameter.set_requires_grad(true);
                }
                model->TokenLinear1->weight.set_requires_grad(true);
            } else if (!full_fine_tuning) {
                for (torch::Tensor parameter : model->parameters())
                    parameter.set_requires_grad(false);
                model->TokenLinear1->weight.set_requires_grad(true);
            }
            std::cout << (full_fine_tuning ? "Full fine-tuning from "
                : acoustic_fine_tuning ? "Acoustic fine-tuning from "
                : identity_fine_tuning ? "Identity fine-tuning from "
                : word_interval_fine_tuning ? "Word-interval fine-tuning from "
                : "Fine-tuning from ")
                << initial_checkpoint << '\n';
        } else {
            std::vector<torch::Tensor> training_audio;
            for (const SequenceCase& sequence : dataset) if (sequence.Split == "train")
                training_audio.push_back(tensors(sequence).Audio);
            torch::Tensor concatenated = torch::cat(training_audio, 2);
            model->AudioMean.copy_(concatenated.mean({0, 2}, true));
            model->AudioScale.copy_(concatenated.std({0, 2}, false, true).clamp_min(1.0e-5));
        }
        model->to(device);
        torch::optim::AdamW optimizer(model->parameters(),
            torch::optim::AdamWOptions(
                (full_fine_tuning || acoustic_fine_tuning || identity_fine_tuning
                    || word_interval_fine_tuning)
                    ? 1.0e-4 : 0.001)
                .weight_decay(fine_tuning ? 0.0 : 5.0e-4));
        std::vector<size_t> training_indices;
        for (size_t i = 0; i < dataset.size(); ++i)
            if (dataset[i].Split == "train") training_indices.push_back(i);
        std::mt19937 random(0x4c49504c);
        const fs::path checkpoint = root / "outputs/runs/latest/neural_streamer.pt";
        const fs::path candidate_directory =
            root / "outputs/runs/neural_streamer_candidates";
        fs::create_directories(candidate_directory);
        write_runtime_checkpoint(candidate_directory / "epoch_00.bin", model);
        double best_validation = validation_loss(model, dataset, device);
        DecodedValidationMetrics best_decoded = validation_decoded_metrics(
            model, dataset, device);
        double best_selection_score = best_decoded.SelectionScore;
        int best_epoch = 0;
        int stale = 0;
        torch::save(model, checkpoint.string());
        const int maximum_epochs = fine_tuning ? 10 : 20;
        for (int epoch = 1; epoch <= maximum_epochs; ++epoch) {
            model->train();
            std::vector<size_t> epoch_indices = training_indices;
            if (fine_tuning || epoch <= 6) {
                for (size_t index : training_indices) {
                    const bool has_region_gap = std::any_of(
                        dataset[index].Tokens.begin(), dataset[index].Tokens.end(), [](const TokenRow& token) {
                            return token.IsSilence > 0.5f
                                && (token.IsRegionStart > 0.5f || token.IsRegionEnd > 0.5f);
                        });
                    if (has_region_gap) epoch_indices.push_back(index);
                }
            }
            if (fine_tuning || epoch <= 6) {
                for (size_t index : training_indices) {
                    const bool has_sentence_boundary = std::any_of(
                        dataset[index].Tokens.begin(), dataset[index].Tokens.end(),
                        [](const TokenRow& token) {
                            return token.IsSentenceBoundary;
                        });
                    if (has_sentence_boundary) {
                        epoch_indices.push_back(index);
                    }
                }
            }
            std::shuffle(epoch_indices.begin(), epoch_indices.end(), random);
            double epoch_loss = 0.0;
            for (size_t index : epoch_indices) {
                CaseTensors row = tensors(dataset[index]);
                const auto output = model->forward(row.Audio.to(device), row.Tokens.to(device));
                const torch::Tensor scores = output.first;
                const torch::Tensor per_frame = torch::nn::functional::cross_entropy(
                    scores, row.Target.to(device),
                    torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kNone));
                const torch::Tensor weights = (epoch <= 4
                    ? row.CurriculumFrameWeights : row.FrameWeights).to(device);
                const torch::Tensor frame_loss = (per_frame * weights).sum() / weights.sum();
                const torch::Tensor region_weights = row.RegionWeights.to(device);
                const torch::Tensor region_loss =
                    torch::nn::functional::binary_cross_entropy_with_logits(
                        output.second, row.RegionTarget.to(device),
                        torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions()
                            .weight(region_weights).reduction(torch::kSum))
                    / region_weights.sum();
                // Rotate the expensive full-lattice objective across cases so
                // every training utterance receives it once in 16 epochs,
                // while dense occupancy supervision remains present each pass.
                const bool apply_forward_sum =
                    (fnv1a(dataset[index].CaseId) + epoch) % kForwardSumCadence == 0;
                torch::Tensor loss = frame_loss + 0.45f * region_loss;
                loss = loss + kWordIntervalLossWeight * word_interval_loss(
                    scores, row.Target.to(device), dataset[index].Tokens);
                if (apply_forward_sum)
                    loss = loss + 0.25f * forward_sum_loss(scores, dataset[index].Tokens);
                optimizer.zero_grad();
                loss.backward();
                if (word_interval_fine_tuning) {
                    torch::Tensor gradient = model->TokenLinear1->weight.grad();
                    // Adapt only continuous timing, silence, word-boundary,
                    // and punctuation semantics. Phone and pose embeddings
                    // remain bit-for-bit unchanged so interval repair cannot
                    // trade away transcript-owned viseme identity.
                    gradient.slice(1, 0,
                        kPhoneBuckets + kPoseBuckets).zero_();
                } else if (acoustic_fine_tuning || identity_fine_tuning) {
                    torch::Tensor gradient = model->TokenLinear1->weight.grad();
                    // Preserve timing, word-boundary, and punctuation behavior.
                    // Only phone/pose identity columns may adapt alongside the
                    // acoustic encoder; the independent region network remains
                    // bit-for-bit identical to the accepted checkpoint.
                    gradient.slice(1, kPhoneBuckets + kPoseBuckets,
                        kTokenDimensions).zero_();
                } else if (fine_tuning && !full_fine_tuning) {
                    torch::Tensor gradient = model->TokenLinear1->weight.grad();
                    // Preserve every accepted schema-v4 weight. Only the new
                    // punctuation-presence and one-hot type columns may learn.
                    gradient.slice(1, 0,
                        kTokenDimensions - kPunctuationFeatureCount).zero_();
                }
                torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
                optimizer.step();
                epoch_loss += loss.item<double>();
            }
            const double validation = validation_loss(model, dataset, device);
            const DecodedValidationMetrics decoded = validation_decoded_metrics(
                model, dataset, device);
            write_runtime_checkpoint(candidate_directory /
                ("epoch_" + (epoch < 10 ? std::string("0") : std::string())
                    + std::to_string(epoch) + ".bin"), model);
            std::cout << "epoch=" << epoch << " train_ce=" << epoch_loss / epoch_indices.size()
                << " validation_loss=" << validation
                << " decoded_score=" << decoded.SelectionScore
                << " decoded_onset_ms=" << decoded.MeanOnsetAbsErrorMs
                << " decoded_exit_ms=" << decoded.MeanExitAbsErrorMs
                << " decoded_coverage=" << decoded.MeanCoverage
                << " decoded_zero_overlap=" << decoded.ZeroOverlapWords << '\n';
            if (decoded.SelectionScore + 1.0e-4 < best_selection_score) {
                best_validation = validation;
                best_decoded = decoded;
                best_selection_score = decoded.SelectionScore;
                best_epoch = epoch;
                stale = 0;
                torch::save(model, checkpoint.string());
            } else if (++stale >= 4) break;
        }
        torch::load(model, checkpoint.string());
        model->to(device);
        const fs::path runtime_checkpoint =
            root / "outputs/runs/latest/neural_streamer_cuda.bin";
        write_runtime_checkpoint(runtime_checkpoint, model);
        const RuntimeBenchmark runtime = benchmark_runtime(
            runtime_checkpoint, model, dataset, device);
        if (runtime.P95Microseconds > 1000.0)
            throw std::runtime_error("CUDA runtime exceeds the 1 ms p95 budget");
        const fs::path report = root / "outputs/runs/latest/neural_streamer_report.json";
        const std::string fine_tuning_mode = !fine_tuning ? "from_scratch"
            : full_fine_tuning ? "full"
            : acoustic_fine_tuning ? "acoustic"
            : identity_fine_tuning ? "identity"
            : word_interval_fine_tuning ? "word_interval"
            : "new_token_features";
        std::ofstream out(report);
        out << std::fixed << std::setprecision(6)
            << "{\n  \"schema_version\": 5,\n  \"device\": \"" << device << "\",\n"
            << "  \"utterance_count\": " << dataset.size() << ",\n"
            << "  \"initial_checkpoint\": \""
            << (fine_tuning ? initial_checkpoint.generic_string() : std::string()) << "\",\n"
            << "  \"fine_tuning\": " << (fine_tuning ? "true" : "false") << ",\n"
            << "  \"fine_tuning_mode\": \"" << fine_tuning_mode << "\",\n"
            << "  \"fine_tuning_new_token_features_only\": "
            << (fine_tuning_mode == "new_token_features" ? "true" : "false") << ",\n"
            << "  \"interpolated_phone_frame_weight\": "
            << kInterpolatedPhoneWeight << ",\n"
            << "  \"selected_epoch\": " << best_epoch << ",\n"
            << "  \"validation_cross_entropy\": " << best_validation << ",\n"
            << "  \"validation_decoded_selection_score\": "
            << best_selection_score << ",\n"
            << "  \"validation_decoded_word_count\": " << best_decoded.Words << ",\n"
            << "  \"validation_decoded_zero_overlap_words\": "
            << best_decoded.ZeroOverlapWords << ",\n"
            << "  \"validation_decoded_onset_mae_ms\": "
            << best_decoded.MeanOnsetAbsErrorMs << ",\n"
            << "  \"validation_decoded_exit_mae_ms\": "
            << best_decoded.MeanExitAbsErrorMs << ",\n"
            << "  \"validation_decoded_spoken_coverage\": "
            << best_decoded.MeanCoverage << ",\n"
            << "  \"word_interval_loss_weight\": "
            << kWordIntervalLossWeight << ",\n"
            << "  \"forward_sum_weight\": 0.25,\n"
            << "  \"forward_sum_cadence\": " << kForwardSumCadence << ",\n"
            << "  \"streaming_lag_ms\": " << kStreamingLagFrames * kFrameMilliseconds << ",\n"
            << "  \"boundary_curriculum_epochs\": 6,\n"
            << "  \"boundary_band_radius_ms\": 40,\n"
            << "  \"feature_noise_stddev\": 0.015,\n"
            << "  \"audio_feature_count\": " << kAudioDimensions << ",\n"
            << "  \"detector_runtime_feature_count\": 0,\n"
            << "  \"region_head\": {\n"
            << "    \"speech_threshold\": 0.45,\n"
            << "    \"minimum_speech_ms\": 20,\n"
            << "    \"minimum_pause_ms\": 120,\n"
            << "    \"causal_context_ms\": 100,\n"
            << "    \"occupancy_gate_scale\": 2.0\n"
            << "  },\n"
            << "  \"runtime\": {\n"
            << "    \"backend\": \"fused_cuda_fp16\",\n"
            << "    \"chunk_frames\": "
                << offgridai::neural_streamer::kRuntimeChunkFrames << ",\n"
            << "    \"forward_token_window\": "
                << offgridai::neural_streamer::kForwardTokenWindow << ",\n"
            << "    \"benchmark_calls\": " << runtime.Calls << ",\n"
            << "    \"mean_us\": " << runtime.MeanMicroseconds << ",\n"
            << "    \"median_us\": " << runtime.MedianMicroseconds << ",\n"
            << "    \"p95_us\": " << runtime.P95Microseconds << ",\n"
            << "    \"max_us\": " << runtime.MaxMicroseconds << ",\n"
            << "    \"token_setup_us\": " << runtime.TokenSetupMicroseconds << ",\n"
            << "    \"max_logit_error\": " << runtime.MaxLogitError << ",\n"
            << "    \"checkpoint_bytes\": " << runtime.Footprint.CheckpointBytes << ",\n"
            << "    \"resident_weight_bytes\": " << runtime.Footprint.WeightBytes << ",\n"
            << "    \"resident_token_bytes\": " << runtime.Footprint.TokenBytes << ",\n"
            << "    \"stream_priority\": " << runtime.Footprint.StreamPriority << "\n"
            << "  }\n}\n";
        std::cout << "Selected epoch " << best_epoch
            << " decoded score " << best_selection_score
            << " validation loss " << best_validation << '\n';
        std::cout << "CUDA runtime p95 " << runtime.P95Microseconds << " us, weights "
            << runtime.Footprint.WeightBytes << " bytes, max logit error "
            << runtime.MaxLogitError << '\n';
        std::cout << "Wrote " << runtime_checkpoint << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
