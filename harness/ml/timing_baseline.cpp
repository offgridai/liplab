#include "OffgridAITimingModelContract.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
using offgridai::timing_ml::kAudioFeatureCount;
using offgridai::timing_ml::kAudioHistoryFrames;

struct Sample {
    std::string CaseId;
    std::string Split;
    int EventIndex = -1;
    std::vector<float> Scheduler;
    std::vector<float> Audio;
    float Target = 0.0f;
    float RawTargetMs = 0.0f;
    float Actionable = 0.0f;
};

struct LinearModel {
    std::vector<float> Mean;
    std::vector<float> Scale;
    std::vector<float> Weight;
    float Bias = 0.0f;
};

struct Metrics {
    int Rows = 0;
    double ClampedMaeMs = 0.0;
    double RawBeforeMaeMs = 0.0;
    double RawAfterMaeMs = 0.0;
    double BoundedOracleMaeMs = 0.0;
    double BeneficialRate = 0.0;
};

static std::vector<std::string> parse_csv_row(const std::string& line)
{
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char ch = line[i];
        if (ch == '"') {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
                field.push_back('"');
                ++i;
            } else {
                quoted = !quoted;
            }
        } else if (ch == ',' && !quoted) {
            fields.push_back(field);
            field.clear();
        } else {
            field.push_back(ch);
        }
    }
    fields.push_back(field);
    return fields;
}

static std::string json_string_value(const std::string& line)
{
    const size_t colon = line.find(':');
    const size_t first = line.find('"', colon + 1);
    const size_t last = line.find('"', first + 1);
    if (colon == std::string::npos || first == std::string::npos || last == std::string::npos) {
        throw std::runtime_error("invalid split manifest line: " + line);
    }
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
        if (line.find("\"case_id\"") != std::string::npos) {
            pending_case = json_string_value(line);
        } else if (!pending_case.empty() && line.find("\"split\"") != std::string::npos) {
            result[pending_case] = json_string_value(line);
            pending_case.clear();
        }
    }
    if (result.empty()) throw std::runtime_error("split manifest contained no cases");
    return result;
}

static std::uint32_t fnv1a(const std::string& value)
{
    std::uint32_t hash = 2166136261u;
    for (unsigned char ch : value) {
        hash ^= ch;
        hash *= 16777619u;
    }
    return hash;
}

static float number(const std::vector<std::string>& row, const std::map<std::string, size_t>& columns,
    const std::string& name)
{
    const auto found = columns.find(name);
    if (found == columns.end()) throw std::runtime_error("missing dataset column: " + name);
    return std::stof(row.at(found->second));
}

static const std::string& text(const std::vector<std::string>& row,
    const std::map<std::string, size_t>& columns, const std::string& name)
{
    const auto found = columns.find(name);
    if (found == columns.end()) throw std::runtime_error("missing dataset column: " + name);
    return row.at(found->second);
}

static std::vector<float> scheduler_features(const std::vector<std::string>& row,
    const std::map<std::string, size_t>& columns)
{
    const float playback = number(row, columns, "commit_playback_sec");
    const float anchor = number(row, columns, "acoustic_anchor_sec");
    const float region_start = number(row, columns, "active_region_start_sec");
    const float region_end = number(row, columns, "active_region_end_sec");
    const float last_match = number(row, columns, "last_matched_audio_sec");
    std::vector<float> result = {
        number(row, columns, "phone_weight_sec"),
        number(row, columns, "is_vowel"),
        number(row, columns, "visual_role"),
        number(row, columns, "prior_center_sec") - playback,
        number(row, columns, "lead_adjusted_center_sec") - playback,
        number(row, columns, "deterministic_center_sec") - playback,
        number(row, columns, "commit_lead_sec"),
        number(row, columns, "playback_offset_sec"),
        number(row, columns, "total_paused_sec"),
        number(row, columns, "total_center_delay_sec"),
        number(row, columns, "mapped_to_observed_speech"),
        number(row, columns, "used_initial_anchor"),
        number(row, columns, "used_resume_anchor"),
        anchor >= 0.0f ? anchor - playback : -1.0f,
        number(row, columns, "acoustic_anchor_error_sec"),
        number(row, columns, "timeline_rate"),
        region_start >= 0.0f ? region_start - playback : -1.0f,
        region_end >= 0.0f ? region_end - playback : -1.0f,
        last_match >= 0.0f ? last_match - playback : -1.0f,
        number(row, columns, "last_matched_confidence"),
        number(row, columns, "audio_valid_frames") / static_cast<float>(kAudioHistoryFrames),
    };

    constexpr int phone_buckets = 32;
    constexpr int pose_buckets = 16;
    result.resize(result.size() + phone_buckets + pose_buckets, 0.0f);
    result[21 + fnv1a(text(row, columns, "phone_base")) % phone_buckets] = 1.0f;
    result[21 + phone_buckets + fnv1a(text(row, columns, "pose")) % pose_buckets] = 1.0f;
    return result;
}

static std::vector<Sample> read_dataset(const fs::path& run_root,
    const std::map<std::string, std::string>& splits)
{
    std::vector<Sample> samples;
    for (const auto& assignment : splits) {
        const fs::path csv_path = run_root / assignment.first / "timing_training_rows.csv";
        if (!fs::exists(csv_path)) throw std::runtime_error("missing dataset rows: " + csv_path.string());
        std::ifstream input(csv_path);
        std::string line;
        if (!std::getline(input, line)) throw std::runtime_error("empty dataset: " + csv_path.string());
        const std::vector<std::string> header = parse_csv_row(line);
        std::map<std::string, size_t> columns;
        for (size_t i = 0; i < header.size(); ++i) columns[header[i]] = i;
        std::vector<size_t> audio_columns;
        for (int frame = 0; frame < kAudioHistoryFrames; ++frame) {
            std::ostringstream prefix;
            prefix << "audio_" << std::setw(2) << std::setfill('0') << frame << '_';
            for (size_t i = 0; i < header.size(); ++i) {
                if (header[i].rfind(prefix.str(), 0) == 0) audio_columns.push_back(i);
            }
        }
        if (audio_columns.size() != static_cast<size_t>(kAudioFeatureCount * kAudioHistoryFrames)) {
            throw std::runtime_error("audio history does not match model contract: " + csv_path.string());
        }
        while (std::getline(input, line)) {
            if (line.empty()) continue;
            const std::vector<std::string> row = parse_csv_row(line);
            if (row.size() != header.size()) throw std::runtime_error("malformed CSV row: " + csv_path.string());
            Sample sample;
            sample.CaseId = assignment.first;
            sample.Split = assignment.second;
            sample.EventIndex = static_cast<int>(number(row, columns, "event_index"));
            sample.Scheduler = scheduler_features(row, columns);
            sample.Audio.reserve(audio_columns.size());
            for (size_t column : audio_columns) sample.Audio.push_back(std::stof(row[column]));
            sample.Target = number(row, columns, "target_shift_clamped_100_ms") / 100.0f;
            sample.RawTargetMs = number(row, columns, "target_shift_ms");
            sample.Actionable = number(row, columns, "target_actionable");
            samples.push_back(std::move(sample));
        }
    }
    return samples;
}

static std::uint8_t split_id(const std::string& split)
{
    if (split == "train") return 0;
    if (split == "validation") return 1;
    if (split == "text_test") return 2;
    if (split == "speaker_holdout") return 3;
    throw std::runtime_error("unknown dataset split: " + split);
}

template <typename T>
static void write_binary(std::ofstream& output, const T& value)
{
    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

static void write_tensor_dataset(const fs::path& output_path, const fs::path& index_path,
    const std::vector<Sample>& samples)
{
    if (samples.empty()) throw std::runtime_error("cannot export an empty tensor dataset");
    std::ofstream output(output_path, std::ios::binary);
    if (!output) throw std::runtime_error("cannot write tensor dataset: " + output_path.string());
    std::ofstream index(index_path);
    if (!index) throw std::runtime_error("cannot write tensor index: " + index_path.string());
    index << "split_id,split_row,case_id,event_index\n";
    std::array<std::uint32_t, 4> split_rows{};
    const char magic[8] = {'L', 'I', 'P', 'T', 'D', 'S', '1', '\0'};
    output.write(magic, sizeof(magic));
    const std::uint32_t version = 1;
    const std::uint32_t row_count = static_cast<std::uint32_t>(samples.size());
    const std::uint32_t scheduler_dimensions = static_cast<std::uint32_t>(samples.front().Scheduler.size());
    const std::uint32_t audio_dimensions = static_cast<std::uint32_t>(samples.front().Audio.size());
    write_binary(output, version);
    write_binary(output, row_count);
    write_binary(output, scheduler_dimensions);
    write_binary(output, audio_dimensions);
    for (const Sample& sample : samples) {
        if (sample.Scheduler.size() != scheduler_dimensions || sample.Audio.size() != audio_dimensions) {
            throw std::runtime_error("inconsistent tensor dimensions");
        }
        const std::uint8_t split = split_id(sample.Split);
        index << static_cast<int>(split) << ',' << split_rows[split]++ << ','
            << sample.CaseId << ',' << sample.EventIndex << '\n';
        const std::uint8_t padding[3] = {};
        write_binary(output, split);
        output.write(reinterpret_cast<const char*>(padding), sizeof(padding));
        write_binary(output, sample.Target);
        write_binary(output, sample.RawTargetMs);
        write_binary(output, sample.Actionable);
        output.write(reinterpret_cast<const char*>(sample.Scheduler.data()),
            static_cast<std::streamsize>(sample.Scheduler.size() * sizeof(float)));
        output.write(reinterpret_cast<const char*>(sample.Audio.data()),
            static_cast<std::streamsize>(sample.Audio.size() * sizeof(float)));
    }
    if (!output) throw std::runtime_error("failed while writing tensor dataset: " + output_path.string());
}

static std::vector<float> select_features(const Sample& sample, const std::string& ablation)
{
    if (ablation == "scheduler") return sample.Scheduler;
    if (ablation == "audio") return sample.Audio;
    std::vector<float> result = sample.Scheduler;
    result.insert(result.end(), sample.Audio.begin(), sample.Audio.end());
    return result;
}

static LinearModel train(const std::vector<Sample>& samples, const std::string& ablation)
{
    std::vector<size_t> train_rows;
    for (size_t i = 0; i < samples.size(); ++i) if (samples[i].Split == "train") train_rows.push_back(i);
    if (train_rows.empty()) throw std::runtime_error("training split is empty");
    const size_t dimensions = select_features(samples[train_rows.front()], ablation).size();
    LinearModel model;
    model.Mean.assign(dimensions, 0.0f);
    model.Scale.assign(dimensions, 0.0f);
    model.Weight.assign(dimensions, 0.0f);
    for (size_t row_index : train_rows) {
        const auto features = select_features(samples[row_index], ablation);
        for (size_t i = 0; i < dimensions; ++i) model.Mean[i] += features[i];
    }
    for (float& value : model.Mean) value /= static_cast<float>(train_rows.size());
    for (size_t row_index : train_rows) {
        const auto features = select_features(samples[row_index], ablation);
        for (size_t i = 0; i < dimensions; ++i) {
            const float delta = features[i] - model.Mean[i];
            model.Scale[i] += delta * delta;
        }
    }
    for (float& value : model.Scale) {
        value = std::sqrt(value / static_cast<float>(train_rows.size()));
        if (value < 1.0e-5f) value = 1.0f;
    }

    std::mt19937 random(0x4c49504c);
    constexpr int epochs = 32;
    constexpr float base_learning_rate = 0.004f;
    constexpr float l2 = 0.00002f;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(train_rows.begin(), train_rows.end(), random);
        const float learning_rate = base_learning_rate / (1.0f + 0.08f * epoch);
        for (size_t row_index : train_rows) {
            const auto features = select_features(samples[row_index], ablation);
            float prediction = model.Bias;
            for (size_t i = 0; i < dimensions; ++i) {
                prediction += model.Weight[i] * (features[i] - model.Mean[i]) / model.Scale[i];
            }
            const float error = prediction - samples[row_index].Target;
            const float gradient = std::clamp(error, -0.25f, 0.25f);
            model.Bias -= learning_rate * gradient;
            for (size_t i = 0; i < dimensions; ++i) {
                const float normalized = (features[i] - model.Mean[i]) / model.Scale[i];
                model.Weight[i] -= learning_rate * (gradient * normalized + l2 * model.Weight[i]);
            }
        }
    }
    return model;
}

static float predict(const LinearModel& model, const std::vector<float>& features)
{
    float value = model.Bias;
    for (size_t i = 0; i < features.size(); ++i) {
        value += model.Weight[i] * (features[i] - model.Mean[i]) / model.Scale[i];
    }
    return std::clamp(value, -1.0f, 1.0f);
}

static Metrics evaluate(const std::vector<Sample>& samples, const std::string& split,
    const std::string& ablation, const LinearModel& model)
{
    Metrics result;
    int beneficial = 0;
    for (const Sample& sample : samples) {
        if (sample.Split != split) continue;
        const float correction_ms = 100.0f * predict(model, select_features(sample, ablation));
        const float clamped_target_ms = 100.0f * sample.Target;
        result.Rows++;
        result.ClampedMaeMs += std::abs(correction_ms - clamped_target_ms);
        result.RawBeforeMaeMs += std::abs(sample.RawTargetMs);
        result.RawAfterMaeMs += std::abs(sample.RawTargetMs - correction_ms);
        result.BoundedOracleMaeMs += std::abs(sample.RawTargetMs - clamped_target_ms);
        beneficial += std::abs(sample.RawTargetMs - correction_ms) < std::abs(sample.RawTargetMs);
    }
    if (result.Rows > 0) {
        result.ClampedMaeMs /= result.Rows;
        result.RawBeforeMaeMs /= result.Rows;
        result.RawAfterMaeMs /= result.Rows;
        result.BoundedOracleMaeMs /= result.Rows;
        result.BeneficialRate = static_cast<double>(beneficial) / result.Rows;
    }
    return result;
}

static void write_report(const fs::path& output, const std::map<std::string, std::map<std::string, Metrics>>& report)
{
    std::ofstream out(output);
    out << "{\n  \"schema_version\": 1,\n  \"models\": {\n";
    size_t model_index = 0;
    for (const auto& model : report) {
        out << "    \"" << model.first << "\": {\n";
        size_t split_index = 0;
        for (const auto& split : model.second) {
            const Metrics& m = split.second;
            out << "      \"" << split.first << "\": {\"rows\": " << m.Rows
                << ", \"clamped_target_mae_ms\": " << m.ClampedMaeMs
                << ", \"raw_before_mae_ms\": " << m.RawBeforeMaeMs
                << ", \"raw_after_mae_ms\": " << m.RawAfterMaeMs
                << ", \"bounded_oracle_mae_ms\": " << m.BoundedOracleMaeMs
                << ", \"beneficial_rate\": " << m.BeneficialRate << "}"
                << (++split_index == model.second.size() ? "\n" : ",\n");
        }
        out << "    }" << (++model_index == report.size() ? "\n" : ",\n");
    }
    out << "  }\n}\n";
}

int main(int argc, char** argv)
{
    try {
        const fs::path root = argc > 1 ? fs::absolute(argv[1]) : fs::current_path();
        const fs::path run_root = root / "outputs" / "runs" / "latest";
        const fs::path split_path = root / "inputs" / "speech_library" / "timing_split_v1.json";
        const auto samples = read_dataset(run_root, read_splits(split_path));
        const std::vector<std::string> splits = {"train", "validation", "text_test", "speaker_holdout"};
        const std::vector<std::string> ablations = {"scheduler", "audio", "fused"};
        std::map<std::string, std::map<std::string, Metrics>> report;
        std::cout << std::fixed << std::setprecision(2);
        for (const std::string& ablation : ablations) {
            const LinearModel model = train(samples, ablation);
            std::cout << ablation << " (" << model.Weight.size() << " features)\n";
            for (const std::string& split : splits) {
                const Metrics metrics = evaluate(samples, split, ablation, model);
                report[ablation][split] = metrics;
                std::cout << "  " << split << ": rows=" << metrics.Rows
                    << " target_mae=" << metrics.ClampedMaeMs
                    << " raw=" << metrics.RawBeforeMaeMs << " -> " << metrics.RawAfterMaeMs
                    << " beneficial=" << 100.0 * metrics.BeneficialRate << "%\n";
            }
        }
        const fs::path output = run_root / "timing_linear_baselines.json";
        write_report(output, report);
        std::cout << "Wrote " << output.string() << '\n';
        const fs::path tensor_output = run_root / "timing_tensor_dataset.bin";
        const fs::path tensor_index = run_root / "timing_tensor_index.csv";
        write_tensor_dataset(tensor_output, tensor_index, samples);
        std::cout << "Wrote " << tensor_output.string() << '\n';
        std::cout << "Wrote " << tensor_index.string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
