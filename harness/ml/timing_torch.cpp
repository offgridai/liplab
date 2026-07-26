#include "OffgridAITimingModelContract.h"

#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace fs = std::filesystem;
using offgridai::timing_ml::kAudioFeatureCount;
using offgridai::timing_ml::kAudioHistoryFrames;

struct TensorRows {
    torch::Tensor Scheduler;
    torch::Tensor Audio;
    torch::Tensor Target;
    torch::Tensor RawTargetMs;
    torch::Tensor Actionable;
    std::vector<std::string> CaseIds;
    std::vector<int> EventIndices;
};

struct EvaluationMetrics {
    std::int64_t Rows = 0;
    double TargetMaeMs = 0.0;
    double RawBeforeMaeMs = 0.0;
    double RawAfterMaeMs = 0.0;
    double GatedRawAfterMaeMs = 0.0;
    double AcceptanceRate = 0.0;
    double ConfidenceAccuracy = 0.0;
};

template <typename T>
static T read_binary(std::ifstream& input)
{
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}

static std::vector<TensorRows> read_tensor_dataset(const fs::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read tensor dataset: " + path.string());
    char magic[8]{};
    input.read(magic, sizeof(magic));
    if (std::string(magic) != "LIPTDS1") throw std::runtime_error("invalid tensor dataset magic");
    const std::uint32_t version = read_binary<std::uint32_t>(input);
    const std::uint32_t rows = read_binary<std::uint32_t>(input);
    const std::uint32_t scheduler_dimensions = read_binary<std::uint32_t>(input);
    const std::uint32_t audio_dimensions = read_binary<std::uint32_t>(input);
    if (version != 1 || audio_dimensions != kAudioFeatureCount * kAudioHistoryFrames) {
        throw std::runtime_error("unsupported tensor dataset contract");
    }

    struct SplitBuffer {
        std::vector<float> Scheduler;
        std::vector<float> Audio;
        std::vector<float> Target;
        std::vector<float> Raw;
        std::vector<float> Actionable;
    };
    std::vector<SplitBuffer> buffers(4);
    for (std::uint32_t row = 0; row < rows; ++row) {
        const std::uint8_t split = read_binary<std::uint8_t>(input);
        char padding[3];
        input.read(padding, sizeof(padding));
        const float target = read_binary<float>(input);
        const float raw = read_binary<float>(input);
        const float actionable = read_binary<float>(input);
        if (split >= buffers.size()) throw std::runtime_error("invalid tensor split id");
        SplitBuffer& buffer = buffers[split];
        const size_t scheduler_offset = buffer.Scheduler.size();
        const size_t audio_offset = buffer.Audio.size();
        buffer.Scheduler.resize(scheduler_offset + scheduler_dimensions);
        buffer.Audio.resize(audio_offset + audio_dimensions);
        input.read(reinterpret_cast<char*>(buffer.Scheduler.data() + scheduler_offset),
            static_cast<std::streamsize>(scheduler_dimensions * sizeof(float)));
        input.read(reinterpret_cast<char*>(buffer.Audio.data() + audio_offset),
            static_cast<std::streamsize>(audio_dimensions * sizeof(float)));
        buffer.Target.push_back(target);
        buffer.Raw.push_back(raw);
        buffer.Actionable.push_back(actionable);
    }
    if (!input) throw std::runtime_error("truncated tensor dataset");

    std::vector<TensorRows> result;
    for (SplitBuffer& buffer : buffers) {
        const auto count = static_cast<std::int64_t>(buffer.Target.size());
        TensorRows split;
        split.Scheduler = torch::from_blob(buffer.Scheduler.data(), {count, scheduler_dimensions}).clone();
        split.Audio = torch::from_blob(buffer.Audio.data(),
            {count, kAudioHistoryFrames, kAudioFeatureCount}).clone().transpose(1, 2).contiguous();
        split.Target = torch::from_blob(buffer.Target.data(), {count}).clone();
        split.RawTargetMs = torch::from_blob(buffer.Raw.data(), {count}).clone();
        split.Actionable = torch::from_blob(buffer.Actionable.data(), {count}).clone();
        result.push_back(std::move(split));
    }
    return result;
}

static void read_tensor_index(const fs::path& path, std::vector<TensorRows>& splits)
{
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot read tensor index: " + path.string());
    std::string line;
    std::getline(input, line);
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        std::stringstream row(line);
        std::string split_text;
        std::string split_row_text;
        std::string case_id;
        std::string event_text;
        std::getline(row, split_text, ',');
        std::getline(row, split_row_text, ',');
        std::getline(row, case_id, ',');
        std::getline(row, event_text, ',');
        const size_t split = static_cast<size_t>(std::stoul(split_text));
        const size_t split_row = static_cast<size_t>(std::stoul(split_row_text));
        if (split >= splits.size() || split_row >= static_cast<size_t>(splits[split].Target.size(0))) {
            throw std::runtime_error("tensor index row is out of range");
        }
        TensorRows& rows = splits[split];
        if (rows.CaseIds.empty()) {
            rows.CaseIds.resize(static_cast<size_t>(rows.Target.size(0)));
            rows.EventIndices.resize(static_cast<size_t>(rows.Target.size(0)), -1);
        }
        rows.CaseIds[split_row] = case_id;
        rows.EventIndices[split_row] = std::stoi(event_text);
    }
    for (const TensorRows& rows : splits) {
        if (rows.CaseIds.size() != static_cast<size_t>(rows.Target.size(0))
            || std::find(rows.EventIndices.begin(), rows.EventIndices.end(), -1) != rows.EventIndices.end()) {
            throw std::runtime_error("tensor index does not cover every row");
        }
    }
}

struct TimingNetworkImpl : torch::nn::Module {
    TimingNetworkImpl(std::int64_t scheduler_dimensions)
    {
        AudioEncoder = register_module("audio_encoder", torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(kAudioFeatureCount, 32, 3).padding(1)),
            torch::nn::ReLU(),
            torch::nn::Conv1d(torch::nn::Conv1dOptions(32, 48, 3).padding(1)),
            torch::nn::ReLU()));
        SchedulerEncoder = register_module("scheduler_encoder", torch::nn::Sequential(
            torch::nn::Linear(scheduler_dimensions, 32), torch::nn::ReLU(), torch::nn::Dropout(0.10)));
        Fusion = register_module("fusion", torch::nn::Sequential(
            torch::nn::Linear(80, 64), torch::nn::ReLU(), torch::nn::Dropout(0.15),
            torch::nn::Linear(64, 32), torch::nn::ReLU()));
        DeltaHead = register_module("delta_head", torch::nn::Linear(32, 1));
        ConfidenceHead = register_module("confidence_head", torch::nn::Linear(32, 1));
        SchedulerMean = register_buffer("scheduler_mean", torch::zeros({1, scheduler_dimensions}));
        SchedulerScale = register_buffer("scheduler_scale", torch::ones({1, scheduler_dimensions}));
        AudioMean = register_buffer("audio_mean", torch::zeros({1, kAudioFeatureCount, 1}));
        AudioScale = register_buffer("audio_scale", torch::ones({1, kAudioFeatureCount, 1}));
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor audio, torch::Tensor scheduler)
    {
        audio = (audio - AudioMean) / AudioScale;
        scheduler = (scheduler - SchedulerMean) / SchedulerScale;
        torch::Tensor audio_state = AudioEncoder->forward(audio).mean(2);
        torch::Tensor scheduler_state = SchedulerEncoder->forward(scheduler);
        torch::Tensor state = Fusion->forward(torch::cat({audio_state, scheduler_state}, 1));
        return {torch::tanh(DeltaHead->forward(state)).squeeze(1),
            ConfidenceHead->forward(state).squeeze(1)};
    }

    torch::nn::Sequential AudioEncoder{nullptr};
    torch::nn::Sequential SchedulerEncoder{nullptr};
    torch::nn::Sequential Fusion{nullptr};
    torch::nn::Linear DeltaHead{nullptr};
    torch::nn::Linear ConfidenceHead{nullptr};
    torch::Tensor SchedulerMean;
    torch::Tensor SchedulerScale;
    torch::Tensor AudioMean;
    torch::Tensor AudioScale;
};
TORCH_MODULE(TimingNetwork);

static EvaluationMetrics evaluate(TimingNetwork& model, const TensorRows& rows,
    const torch::Device& device, const std::string& name)
{
    torch::NoGradGuard no_grad;
    auto [delta, confidence] = model->forward(rows.Audio.to(device), rows.Scheduler.to(device));
    const torch::Tensor correction_ms = 100.0f * delta.cpu();
    const torch::Tensor confidence_probability = torch::sigmoid(confidence).cpu();
    const torch::Tensor confidence_gate = confidence_probability.ge(0.5f).to(torch::kFloat32);
    const torch::Tensor gated_correction_ms = correction_ms * confidence_gate;
    EvaluationMetrics result;
    result.Rows = rows.Target.size(0);
    result.TargetMaeMs = (correction_ms - 100.0f * rows.Target).abs().mean().item<double>();
    result.RawBeforeMaeMs = rows.RawTargetMs.abs().mean().item<double>();
    result.RawAfterMaeMs = (rows.RawTargetMs - correction_ms).abs().mean().item<double>();
    result.GatedRawAfterMaeMs = (rows.RawTargetMs - gated_correction_ms).abs().mean().item<double>();
    result.AcceptanceRate = confidence_gate.mean().item<double>();
    result.ConfidenceAccuracy = (confidence_probability.ge(0.5f)
        == rows.Actionable.ge(0.5f)).to(torch::kFloat32).mean().item<double>();
    std::cout << name << ": target_mae_ms=" << result.TargetMaeMs << " raw=" << result.RawBeforeMaeMs
        << " -> " << result.RawAfterMaeMs << " gated=" << result.GatedRawAfterMaeMs
        << " accepted=" << result.AcceptanceRate
        << " confidence_accuracy=" << result.ConfidenceAccuracy << '\n';
    return result;
}

static void write_report(const fs::path& path, int best_epoch, double best_validation_mae,
    const std::vector<std::string>& names, const std::vector<EvaluationMetrics>& metrics)
{
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write timing report: " + path.string());
    out << std::fixed << std::setprecision(6)
        << "{\n  \"schema_version\": 1,\n"
        << "  \"device\": \"cuda\",\n"
        << "  \"seed\": 1279873100,\n"
        << "  \"selected_epoch\": " << best_epoch << ",\n"
        << "  \"selected_validation_raw_mae_ms\": " << best_validation_mae << ",\n"
        << "  \"splits\": {\n";
    for (size_t i = 0; i < metrics.size(); ++i) {
        const EvaluationMetrics& row = metrics[i];
        out << "    \"" << names[i] << "\": {\"rows\": " << row.Rows
            << ", \"target_mae_ms\": " << row.TargetMaeMs
            << ", \"raw_before_mae_ms\": " << row.RawBeforeMaeMs
            << ", \"raw_after_mae_ms\": " << row.RawAfterMaeMs
            << ", \"gated_raw_after_mae_ms\": " << row.GatedRawAfterMaeMs
            << ", \"acceptance_rate\": " << row.AcceptanceRate
            << ", \"confidence_accuracy\": " << row.ConfidenceAccuracy << "}"
            << (i + 1 == metrics.size() ? "\n" : ",\n");
    }
    out << "  }\n}\n";
}

static double validation_raw_mae(TimingNetwork& model, const TensorRows& rows, const torch::Device& device)
{
    torch::NoGradGuard no_grad;
    model->eval();
    auto [delta, confidence] = model->forward(rows.Audio.to(device), rows.Scheduler.to(device));
    (void)confidence;
    return (rows.RawTargetMs.to(device) - 100.0f * delta).abs().mean().item<double>();
}

static void write_predictions(const fs::path& path, TimingNetwork& model,
    const std::vector<TensorRows>& splits, const std::vector<std::string>& split_names,
    const torch::Device& device)
{
    torch::NoGradGuard no_grad;
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write timing predictions: " + path.string());
    out << "case_id,event_index,split,correction_ms,confidence\n";
    out << std::fixed << std::setprecision(6);
    for (size_t split_index = 0; split_index < splits.size(); ++split_index) {
        const TensorRows& rows = splits[split_index];
        auto [delta, confidence_logits] = model->forward(rows.Audio.to(device), rows.Scheduler.to(device));
        const torch::Tensor correction = (100.0f * delta).cpu();
        const torch::Tensor confidence = torch::sigmoid(confidence_logits).cpu();
        for (std::int64_t row = 0; row < correction.size(0); ++row) {
            out << rows.CaseIds[static_cast<size_t>(row)] << ','
                << rows.EventIndices[static_cast<size_t>(row)] << ','
                << split_names[split_index] << ','
                << correction[row].item<float>() << ','
                << confidence[row].item<float>() << '\n';
        }
    }
}

int main(int argc, char** argv)
{
    try {
        const fs::path root = argc > 1 ? fs::absolute(argv[1]) : fs::current_path();
        const fs::path dataset_path = root / "outputs" / "runs" / "latest" / "timing_tensor_dataset.bin";
        const fs::path dataset_index_path = root / "outputs" / "runs" / "latest" / "timing_tensor_index.csv";
        const fs::path checkpoint = root / "outputs" / "runs" / "latest" / "timing_torch_model.pt";
        std::vector<TensorRows> splits = read_tensor_dataset(dataset_path);
        read_tensor_index(dataset_index_path, splits);
        TensorRows& train = splits[0];
#ifdef _WIN32
        _putenv_s("CUBLAS_WORKSPACE_CONFIG", ":4096:8");
#endif
        torch::manual_seed(0x4c49504c);
        at::globalContext().setDeterministicAlgorithms(true, false);
        const torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        std::cout << "Training timing network on " << device << '\n';
        TimingNetwork model(train.Scheduler.size(1));
        model->SchedulerMean.copy_(train.Scheduler.mean(0, true));
        model->SchedulerScale.copy_(train.Scheduler.std(0, false, true).clamp_min(1.0e-5));
        model->AudioMean.copy_(train.Audio.mean({0, 2}, true));
        model->AudioScale.copy_(train.Audio.std({0, 2}, false, true).clamp_min(1.0e-5));
        model->to(device);
        const torch::Tensor audio = train.Audio.to(device);
        const torch::Tensor scheduler = train.Scheduler.to(device);
        const torch::Tensor target = train.Target.to(device);
        const torch::Tensor actionable = train.Actionable.to(device);
        torch::optim::AdamW optimizer(model->parameters(), torch::optim::AdamWOptions(0.001).weight_decay(5.0e-4));
        constexpr std::int64_t batch_size = 256;
        constexpr int epochs = 100;
        constexpr int patience = 12;
        double best_validation_mae = std::numeric_limits<double>::infinity();
        int best_epoch = -1;
        int stale_epochs = 0;
        for (int epoch = 0; epoch < epochs; ++epoch) {
            model->train();
            const torch::Tensor order = torch::randperm(audio.size(0), torch::TensorOptions().device(device).dtype(torch::kLong));
            for (std::int64_t start = 0; start < audio.size(0); start += batch_size) {
                const torch::Tensor indices = order.slice(0, start, std::min(start + batch_size, audio.size(0)));
                auto [delta, confidence] = model->forward(audio.index_select(0, indices), scheduler.index_select(0, indices));
                const torch::Tensor delta_loss = torch::smooth_l1_loss(delta, target.index_select(0, indices));
                const torch::Tensor confidence_loss = torch::binary_cross_entropy_with_logits(
                    confidence, actionable.index_select(0, indices));
                const torch::Tensor loss = delta_loss + 0.15f * confidence_loss;
                optimizer.zero_grad();
                loss.backward();
                torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
                optimizer.step();
            }
            const double validation_mae = validation_raw_mae(model, splits[1], device);
            if (validation_mae + 0.01 < best_validation_mae) {
                best_validation_mae = validation_mae;
                best_epoch = epoch + 1;
                stale_epochs = 0;
                torch::save(model, checkpoint.string());
            } else if (++stale_epochs >= patience) {
                break;
            }
        }
        torch::load(model, checkpoint.string());
        model->to(device);
        model->eval();
        std::cout << "Selected epoch " << best_epoch << " at validation raw MAE "
            << best_validation_mae << " ms\n";
        const std::vector<std::string> split_names = {"train", "validation", "text_test", "speaker_holdout"};
        std::vector<EvaluationMetrics> evaluation;
        for (size_t i = 0; i < split_names.size(); ++i) {
            evaluation.push_back(evaluate(model, splits[i], device, split_names[i]));
        }
        const fs::path report = root / "outputs" / "runs" / "latest" / "timing_torch_report.json";
        write_report(report, best_epoch, best_validation_mae, split_names, evaluation);
        const fs::path predictions = root / "outputs" / "runs" / "latest" / "timing_torch_predictions.csv";
        write_predictions(predictions, model, splits, split_names, device);
        std::cout << "Wrote " << checkpoint << '\n';
        std::cout << "Wrote " << report << '\n';
        std::cout << "Wrote " << predictions << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
