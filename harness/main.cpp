#include "Lipsync/OffgridAILipsyncRuntimeAdapter.h"
#include "Lipsync/OffgridAIAcousticEvidence.h"
#include "Lipsync/OffgridAIStreamingEvidenceSurface.h"
#include "Lipsync/OffgridAIStreamingSyllablePositionEstimator.h"
#include "Lipsync/OffgridAIVisemePerformer.h"

#include <cstring>
#include <cctype>
#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <iostream>
#include <map>
#include <optional>
#include <numeric>
#include <sstream>
#include <string>
#include <limits>
#include <vector>

namespace fs = std::filesystem;

using FOffgridAIAlignedVisemeTrack = FOffgridAICommittedVisemeTrack;
using FOffgridAIAlignedVisemeEvent = FOffgridAICommittedVisemeEvent;

static std::string derived_pause_family(const FOffgridAIStreamingAudioFeatureFrame& frame)
{
    if (frame.bStrongQuiet || frame.bFrameClosedSpeechRegion)
    {
        if (frame.SilenceAccumSec >= 0.180f)
        {
            return "hard_silence";
        }
        if (frame.SilenceAccumSec >= 0.130f)
        {
            return "speech_region_gap";
        }
        if (frame.SilenceAccumSec >= 0.065f)
        {
            return "word_gap";
        }
        return "micro_gap";
    }

    if (!frame.bInSpeechAfterFrame && frame.bEndpointCandidateActive)
    {
        if (frame.SilenceAccumSec >= 0.130f)
        {
            return "speech_region_gap";
        }
        if (frame.SilenceAccumSec >= 0.020f)
        {
            return "micro_gap";
        }
    }

    return "continuous_speech";
}

static double derived_pause_confidence(const FOffgridAIStreamingAudioFeatureFrame& frame)
{
    const double low_rms = std::max(0.0, std::min(1.0, (0.12 - static_cast<double>(frame.RMSNorm)) / 0.12));
    const double low_evidence = std::max(0.0, std::min(1.0, (0.30 - static_cast<double>(frame.SpeechEvidence)) / 0.30));
    const double silence = std::max(0.0, std::min(1.0, static_cast<double>(frame.SilenceAccumSec) / 0.18));
    const double strong_quiet = frame.bStrongQuiet ? 1.0 : 0.0;
    const double endpoint = frame.bEndpointCandidateActive ? 1.0 : 0.0;
    return std::max(0.0, std::min(1.0,
        low_rms * 0.35 +
        low_evidence * 0.25 +
        silence * 0.20 +
        strong_quiet * 0.15 +
        endpoint * 0.05));
}

struct Wav
{
    int sample_rate = 0;
    int channels = 0;
    std::vector<int16_t> samples;
};

struct HandmadeLabel
{
    double start = 0.0;
    double end = 0.0;
    std::string pose;
    std::string word;
    double confidence = 1.0;
    int word_index = -1;
    int speech_region_index = -1;
    int sentence_index = -1;
    int source_phone_global_index = -1;
    int source_phone_word_index = -1;
    std::string source_phone_base;
};

struct GoldWordTiming
{
    int word_index = -1;
    std::string word;
    double start = 0.0;
    double end = 0.0;
    int speech_region_index = -1;
    int sentence_index = -1;
};

struct GoldPhoneTiming
{
    int global_phone_index = -1;
    int word_phone_index = -1;
    std::string phone;
    std::string phone_base;
    std::string word;
    double start = 0.0;
    double end = 0.0;
    int word_index = -1;
    int speech_region_index = -1;
    int sentence_index = -1;
};

struct GoldSpeechRegion
{
    int index = -1;
    double start = 0.0;
    double end = 0.0;
    double last_speech = 0.0;
};

struct GoldBoundaryTiming
{
    int word_index = -1;
    std::string word;
    int next_word_index = -1;
    std::string next_word;
    std::string mark;
    std::string pause_class;
    bool split_applied = false;
    double direct_word_gap_seconds = 0.0;
    double phone_gap_seconds = 0.0;
    double wave_silence_seconds = 0.0;
    double acoustic_gap_seconds = 0.0;
    int speech_region_index_before = -1;
    int speech_region_index_after = -1;
};

struct StreamingDetectorReport
{
    int pause_frames = 0;
    int pause_matches = 0;
    int gold_word_boundaries = 0;
    int predicted_word_boundaries = 0;
    int matched_word_boundaries = 0;
    double mean_abs_word_boundary_error_ms = 0.0;
    double median_abs_word_boundary_error_ms = 0.0;
    double p90_abs_word_boundary_error_ms = 0.0;
};


struct ProsodicPeakTarget
{
    int target_index = -1;
    int expected_phone_index = -1;
    int global_phone_index = -1;
    int word_index = -1;
    int speech_region_index = -1;
    int stress = 0;
    std::string word;
    std::string phone;
    std::string vowel_family;
    double prior_center = 0.0;
    double gold_start = 0.0;
    double gold_end = 0.0;
    double gold_center = 0.0;
    bool has_gold = false;
};

struct ProsodicPeakObservation
{
    int frame_index = -1;
    int target_index = -1;
    double peak_sec = 0.0;
    double decision_sec = 0.0;
    double prominence = 0.0;
    double family_score = 0.0;
    double score = 0.0;
    double error_ms = 0.0;
    int speech_region_index = -1;
    bool matched = false;
};

struct ProsodicPeakReport
{
    bool available = false;
    int target_count = 0;
    int observation_count = 0;
    int matched_count = 0;
    double precision = 0.0;
    double recall = 0.0;
    double mean_abs_error_ms = 0.0;
    double median_abs_error_ms = 0.0;
    double p90_abs_error_ms = 0.0;
    double mean_decision_latency_ms = 0.0;
    int raw_observation_count = 0;
    int raw_ungradeable_observation_count = 0;
    int raw_matched_count = 0;
    double raw_precision = 0.0;
    double raw_recall = 0.0;
    double raw_mean_abs_error_ms = 0.0;
    std::vector<double> matched_error_samples_ms;
};

struct EvidenceIngredientTarget
{
    std::string type;
    double start = 0.0;
    double end = 0.0;
};

struct EvidenceIngredientTypeReport
{
    int target_count = 0;
    int observation_count = 0;
    int matched_count = 0;
    double precision = 0.0;
    double recall = 0.0;
    double mean_abs_error_ms = 0.0;
    double mean_decision_latency_ms = 0.0;
    int region_count = 0;
    int exact_region_count = 0;
    int region_count_abs_error = 0;
    int clustered_120_target_count = 0;
    int clustered_120_matched_count = 0;
    int clustered_150_target_count = 0;
    int clustered_150_matched_count = 0;
    int clustered_200_target_count = 0;
    int clustered_200_matched_count = 0;
    int outside_speech_observation_count = 0;
};

struct EvidenceIngredientReport
{
    bool available = false;
    double preroll_ms = 0.0;
    double postroll_ms = 0.0;
    std::map<std::string, EvidenceIngredientTypeReport> by_type;
};

struct SyllableCandidateSetReport
{
    bool available = false;
    int matched_pulse_count = 0;
    int unmatched_pulse_count = 0;
    std::array<int, 6> hit_counts{};
};

struct BoundaryAlignmentReport
{
    int reference_count = 0;
    int predicted_count = 0;
    int matched_count = 0;
    int missing_count = 0;
    int extra_count = 0;
    int clean_match_count = 0;
    int merge_count = 0;
    int split_count = 0;
    double mean_abs_start_error_ms = 0.0;
    double median_abs_start_error_ms = 0.0;
    double p90_abs_start_error_ms = 0.0;
    double mean_abs_end_error_ms = 0.0;
    double median_abs_end_error_ms = 0.0;
    double p90_abs_end_error_ms = 0.0;
    double mean_lead_in_ms = 0.0;
    double mean_tail_out_ms = 0.0;
    bool count_mismatch = false;
    std::vector<double> matched_start_error_samples_ms;
    std::vector<double> matched_end_error_samples_ms;
};

struct PauseAlignmentReport
{
    BoundaryAlignmentReport speech_regions;
    BoundaryAlignmentReport text_sentence_spans;
    BoundaryAlignmentReport text_sentence_region_subspans;
};

struct WordOnsetAlignmentReport
{
    int reference_count = 0;
    int matched_count = 0;
    int missing_count = 0;
    double mean_abs_start_error_ms = 0.0;
    double median_abs_start_error_ms = 0.0;
    double p90_abs_start_error_ms = 0.0;
    double mean_early_start_ms = 0.0;
    double mean_late_start_ms = 0.0;
    std::vector<double> matched_start_error_samples_ms;
};

struct IntraWordAlignmentReport
{
    int reference_count = 0;
    int matched_count = 0;
    int missing_count = 0;
    int extra_count = 0;
    double mean_abs_center_error_ms = 0.0;
    double median_abs_center_error_ms = 0.0;
    double p90_abs_center_error_ms = 0.0;
    double mean_abs_start_error_ms = 0.0;
    double mean_abs_end_error_ms = 0.0;
    std::vector<double> matched_center_error_samples_ms;
    std::vector<double> matched_start_error_samples_ms;
};

struct SpeechRegionContainmentReport
{
    int predicted_region_count = 0;
    int event_count = 0;
    int events_with_invalid_region = 0;
    int events_outside_region = 0;
    int events_starting_before_region = 0;
    int events_ending_after_region = 0;
    double mean_early_start_leak_ms = 0.0;
    double mean_late_end_leak_ms = 0.0;
    double max_early_start_leak_ms = 0.0;
    double max_late_end_leak_ms = 0.0;
};

struct DroppedVisemeReport
{
    int dropped_count = 0;
    int dropped_region_closed_count = 0;
    int dropped_missing_region_count = 0;
    int dropped_strong_visible_count = 0;
    double dropped_rate = 0.0;
};

struct TimeSpan
{
    int key = -1;
    double start = 0.0;
    double end = 0.0;
};

struct GradeReport
{
    bool gold_available = true;
    std::string gold_status = "approved";
    int reference_count = 0;
    int committed_count = 0;
    int matched_count = 0;
    int missing_count = 0;
    int extra_count = 0;
    int order_violations = 0;
    double mean_abs_center_error_ms = 0.0;
    double median_abs_center_error_ms = 0.0;
    double p90_abs_center_error_ms = 0.0;
    double mean_abs_start_error_ms = 0.0;
    double mean_abs_end_error_ms = 0.0;
    double mean_abs_duration_error_ms = 0.0;
    double early_start_rate = 0.0;
    double late_tail_rate = 0.0;
    PauseAlignmentReport pause_alignment;
    WordOnsetAlignmentReport word_onset_alignment;
    IntraWordAlignmentReport intra_word_alignment;
    SpeechRegionContainmentReport speech_region_containment;
    DroppedVisemeReport dropped_visemes;
    std::vector<double> matched_center_error_samples_ms;
    std::vector<double> matched_start_error_samples_ms;
};

struct GradeMatch
{
    int label_index = -1;
    int event_index = -1;
};

struct StreamConfig
{
    float buffer_seconds = 0.350f;
    float evidence_postroll_seconds = 1.500f;
    float chunk_seconds = 0.040f;
    float playback_tick_seconds = 0.020f;
};

struct OutputConfig
{
    bool write_detailed_diagnostics = true;
    bool write_monotonic_training_rows = false;
};

struct PredictedNeuralTrackEvent
{
    int token_index = -1;
    int event_index = -1;
    bool is_silence = false;
    double center_sec = 0.0;
    double duration_sec = 0.08;
    double confidence = 0.0;
    bool fallback_used = false;
    int speech_region_index = -1;
    double speech_region_start_sec = -1.0;
    double speech_region_end_sec = -1.0;
};

using PredictedNeuralTrackByCase =
    std::map<std::string, std::vector<PredictedNeuralTrackEvent>>;

struct PredictedNeuralTrackResult
{
    FOffgridAIAlignedVisemeTrack track;
    TArray<FOffgridAIStreamingSpeechRegion> speech_regions;
    int prediction_count = 0;
    int accepted_count = 0;
    int deterministic_fallback_count = 0;
};

static std::string to_std(const FString& value) { return value.Str(); }
static std::string to_std(const FName& value) { return value.Str(); }

static std::string csv_value(const std::string& value)
{
    if (value.find_first_of(",\"\r\n") == std::string::npos)
    {
        return value;
    }
    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (const char ch : value)
    {
        if (ch == '"') escaped.push_back('"');
        escaped.push_back(ch);
    }
    escaped.push_back('"');
    return escaped;
}

static std::vector<std::string> parse_csv_cells(const std::string& line)
{
    std::vector<std::string> cells;
    std::string cell;
    bool in_quotes = false;
    for (size_t i = 0; i < line.size(); ++i)
    {
        const char c = line[i];
        if (c == '"')
        {
            const bool escaped_quote = in_quotes && (i + 1) < line.size() && line[i + 1] == '"';
            if (escaped_quote)
            {
                cell.push_back('"');
                ++i;
            }
            else
            {
                in_quotes = !in_quotes;
            }
            continue;
        }
        if (c == ',' && !in_quotes)
        {
            cells.push_back(cell);
            cell.clear();
            continue;
        }
        cell.push_back(c);
    }
    cells.push_back(cell);
    return cells;
}

static int header_index(const std::vector<std::string>& header, const std::string& name)
{
    for (size_t i = 0; i < header.size(); ++i)
    {
        if (header[i] == name)
        {
            return static_cast<int>(i);
        }
    }
    return -1;
}

static std::string csv_cell(const std::vector<std::string>& cells, int index)
{
    if (index < 0 || static_cast<size_t>(index) >= cells.size())
    {
        return {};
    }
    return cells[static_cast<size_t>(index)];
}

static PredictedNeuralTrackByCase read_predicted_neural_track_csv(const fs::path& path)
{
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot read neural track: " + path.string());
    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("empty neural track: " + path.string());
    const auto header = parse_csv_cells(line);
    const int case_column = header_index(header, "case_id");
    const int token_column = header_index(header, "token_index");
    const int event_column = header_index(header, "event_index");
    const int center_column = header_index(header, "center_sec");
    const int duration_column = header_index(header, "duration_sec");
    const int confidence_column = header_index(header, "confidence");
    const int fallback_column = header_index(header, "fallback_used");
    const int silence_column = header_index(header, "is_silence");
    const int region_index_column = header_index(header, "speech_region_index");
    const int region_start_column = header_index(header, "speech_region_start_sec");
    const int region_end_column = header_index(header, "speech_region_end_sec");
    if (case_column < 0 || token_column < 0 || event_column < 0 || center_column < 0
        || duration_column < 0 || confidence_column < 0 || fallback_column < 0)
    {
        throw std::runtime_error("neural track is missing required columns");
    }
    PredictedNeuralTrackByCase result;
    while (std::getline(input, line))
    {
        if (line.empty()) continue;
        const auto cells = parse_csv_cells(line);
        PredictedNeuralTrackEvent event;
        event.token_index = std::stoi(csv_cell(cells, token_column));
        event.event_index = std::stoi(csv_cell(cells, event_column));
        event.is_silence = csv_cell(cells, silence_column) == "1";
        event.center_sec = std::stod(csv_cell(cells, center_column));
        event.duration_sec = std::stod(csv_cell(cells, duration_column));
        event.confidence = std::stod(csv_cell(cells, confidence_column));
        event.fallback_used = std::stoi(csv_cell(cells, fallback_column)) != 0;
        if (region_index_column >= 0)
            event.speech_region_index = std::stoi(csv_cell(cells, region_index_column));
        if (region_start_column >= 0)
            event.speech_region_start_sec = std::stod(csv_cell(cells, region_start_column));
        if (region_end_column >= 0)
            event.speech_region_end_sec = std::stod(csv_cell(cells, region_end_column));
        result[csv_cell(cells, case_column)].push_back(std::move(event));
    }
    for (auto& [case_id, events] : result)
    {
        (void)case_id;
        std::sort(events.begin(), events.end(), [](const auto& left, const auto& right) {
            return left.token_index < right.token_index;
        });
    }
    return result;
}

static std::string read_text(const fs::path& path)
{
    std::ifstream file(path);
    std::ostringstream out;
    out << file.rdbuf();
    std::string text = out.str();
    if (text.size() >= 3
        && static_cast<unsigned char>(text[0]) == 0xEF
        && static_cast<unsigned char>(text[1]) == 0xBB
        && static_cast<unsigned char>(text[2]) == 0xBF)
    {
        text.erase(0, 3);
    }
    return text;
}

static void write_text(const fs::path& path, const std::string& text)
{
    fs::create_directories(path.parent_path());
    std::ofstream file(path);
    file << text;
}

static std::string json_escape(const std::string& value)
{
    std::string out;
    out.reserve(value.size());
    for (const char ch : value)
    {
        switch (ch)
        {
        case '\\': out += "\\\\"; break;
        case '"': out += "\\\""; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default: out += ch; break;
        }
    }
    return out;
}

static void write_progress_json(
    const fs::path& out_root,
    const std::string& phase,
    int completed_cases,
    int total_cases,
    const std::string& current_case,
    double elapsed_seconds,
    double last_case_seconds = 0.0)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(3);
    out << "{\n"
        << "  \"phase\": \"" << json_escape(phase) << "\",\n"
        << "  \"completed_cases\": " << completed_cases << ",\n"
        << "  \"total_cases\": " << total_cases << ",\n"
        << "  \"current_case\": \"" << json_escape(current_case) << "\",\n"
        << "  \"elapsed_seconds\": " << elapsed_seconds << ",\n"
        << "  \"last_case_seconds\": " << last_case_seconds << "\n"
        << "}\n";
    write_text(out_root / "progress.json", out.str());
}

static double percentile_ms(std::vector<double> values, double percentile)
{
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    percentile = std::clamp(percentile, 0.0, 1.0);
    const double pos = percentile * static_cast<double>(values.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = static_cast<size_t>(std::ceil(pos));
    if (lo == hi) return values[lo];
    const double t = pos - static_cast<double>(lo);
    return values[lo] * (1.0 - t) + values[hi] * t;
}

static double mean_ms(const std::vector<double>& values)
{
    if (values.empty()) return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

template <class T>
static T read_le(const std::vector<uint8_t>& bytes, size_t offset)
{
    T value{};
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    return value;
}

static Wav read_wav(const fs::path& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open wav: " + path.string());

    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)), {});
    if (bytes.size() < 44 || std::memcmp(bytes.data(), "RIFF", 4) != 0 || std::memcmp(bytes.data() + 8, "WAVE", 4) != 0)
    {
        throw std::runtime_error("bad wav: " + path.string());
    }

    Wav wav;
    size_t pos = 12;
    size_t data_offset = 0;
    size_t data_size = 0;
    uint16_t bits = 0;
    uint16_t format = 0;
    while (pos + 8 <= bytes.size())
    {
        char id[5]{};
        std::memcpy(id, bytes.data() + pos, 4);
        const uint32_t size = read_le<uint32_t>(bytes, pos + 4);
        pos += 8;
        if (std::string(id) == "fmt ")
        {
            format = read_le<uint16_t>(bytes, pos);
            wav.channels = read_le<uint16_t>(bytes, pos + 2);
            wav.sample_rate = static_cast<int>(read_le<uint32_t>(bytes, pos + 4));
            bits = read_le<uint16_t>(bytes, pos + 14);
        }
        else if (std::string(id) == "data")
        {
            data_offset = pos;
            data_size = size;
        }
        pos += size + (size & 1);
    }

    if (format != 1 || bits != 16 || data_offset == 0)
    {
        throw std::runtime_error("only PCM16 wav supported: " + path.string());
    }

    const size_t sample_count = data_size / sizeof(int16_t);
    wav.samples.resize(sample_count);
    std::memcpy(wav.samples.data(), bytes.data() + data_offset, sample_count * sizeof(int16_t));
    return wav;
}

static TArray<uint8> to_pcm_bytes(const std::vector<int16_t>& samples)
{
    TArray<uint8> bytes;
    const auto* raw = reinterpret_cast<const uint8*>(samples.data());
    for (size_t i = 0; i < samples.size() * sizeof(int16_t); ++i)
    {
        bytes.Add(raw[i]);
    }
    return bytes;
}

static std::vector<HandmadeLabel> read_handmade_csv(const fs::path& path)
{
    std::vector<HandmadeLabel> out;
    std::ifstream file(path);
    if (!file) return out;

    std::string line;
    std::getline(file, line);
    const std::vector<std::string> header = parse_csv_cells(line);
    const bool has_header = !header.empty() && (header[0] == "start" || header[0] == "pose");
    const int start_col = has_header ? header_index(header, "start") : 0;
    const int end_col = has_header ? header_index(header, "end") : 1;
    const int pose_col = has_header ? header_index(header, "pose") : 2;
    const int word_col = has_header ? header_index(header, "word") : 3;
    int confidence_col = has_header ? header_index(header, "confidence") : 4;
    int word_index_col = has_header ? header_index(header, "word_index") : 5;
    int speech_region_index_col = has_header ? header_index(header, "speech_region_index") : 6;
    int sentence_index_col = has_header ? header_index(header, "text_sentence_index") : 7;

    if (confidence_col < 0 && !has_header)
    {
        confidence_col = 4;
    }
    if (word_index_col < 0 && !has_header)
    {
        word_index_col = 5;
    }
    if (speech_region_index_col < 0 && !has_header)
    {
        speech_region_index_col = 6;
    }
    if (sentence_index_col < 0 && !has_header)
    {
        sentence_index_col = 7;
    }
    if (sentence_index_col < 0 && has_header)
    {
        sentence_index_col = header_index(header, "sentence_index");
    }

    while (std::getline(file, line))
    {
        if (line.empty()) continue;
        const std::vector<std::string> cells = parse_csv_cells(line);
        if (start_col < 0 || end_col < 0 || pose_col < 0 || word_col < 0) continue;
        HandmadeLabel label;
        label.start = std::stod(csv_cell(cells, start_col));
        label.end = std::stod(csv_cell(cells, end_col));
        label.pose = csv_cell(cells, pose_col);
        label.word = csv_cell(cells, word_col);

        const std::string confidence = csv_cell(cells, confidence_col);
        const std::string word_index = csv_cell(cells, word_index_col);
        const std::string speech_region_index = csv_cell(cells, speech_region_index_col);
        const std::string sentence_index = csv_cell(cells, sentence_index_col);

        if (!confidence.empty()) label.confidence = std::stod(confidence);
        if (!word_index.empty()) label.word_index = std::stoi(word_index);
        if (!speech_region_index.empty()) label.speech_region_index = std::stoi(speech_region_index);
        if (!sentence_index.empty()) label.sentence_index = std::stoi(sentence_index);
        out.push_back(label);
    }
    return out;
}

static std::string strip_stress_digits(std::string phone)
{
    phone.erase(
        std::remove_if(
            phone.begin(),
            phone.end(),
            [](unsigned char c)
            {
                return std::isdigit(c) != 0;
            }),
        phone.end());
    return phone;
}


static bool is_pause_lull_type(const std::string& type)
{
    return type == "comma_lull" || type == "pause_lull";
}

static std::vector<GoldPhoneTiming> read_gold_phones_csv(const fs::path& path)
{
    std::vector<GoldPhoneTiming> out;
    std::ifstream file(path);
    if (!file) return out;

    std::string line;
    std::getline(file, line);
    const std::vector<std::string> header = parse_csv_cells(line);
    const bool has_header = !header.empty() && (header[0] == "global_phone_index" || header[0] == "start");
    const int global_phone_index_col = header_index(header, "global_phone_index");
    const int start_col = has_header ? header_index(header, "start") : 0;
    const int end_col = has_header ? header_index(header, "end") : 1;
    const int phone_col = has_header ? header_index(header, "phone") : 2;
    const int word_col = has_header ? header_index(header, "word") : 3;
    const int word_index_col = has_header ? header_index(header, "word_index") : 5;
    int word_phone_index_col = has_header ? header_index(header, "phone_index") : -1;
    if (word_phone_index_col < 0)
    {
        word_phone_index_col = has_header ? header_index(header, "word_phone_index") : 7;
    }
    const int speech_region_index_col = has_header ? header_index(header, "speech_region_index") : 8;
    int sentence_index_col = has_header ? header_index(header, "text_sentence_index") : 9;
    if (sentence_index_col < 0 && has_header)
    {
        sentence_index_col = header_index(header, "sentence_index");
    }
    int global_index = 0;
    while (std::getline(file, line))
    {
        if (line.empty()) continue;
        const std::vector<std::string> cells = parse_csv_cells(line);
        if (start_col < 0 || end_col < 0 || phone_col < 0 || word_col < 0) continue;

        GoldPhoneTiming phone;
        if (global_phone_index_col >= 0)
        {
            const std::string global_phone_index = csv_cell(cells, global_phone_index_col);
            phone.global_phone_index = global_phone_index.empty() ? global_index : std::stoi(global_phone_index);
        }
        else
        {
            phone.global_phone_index = global_index;
        }

        phone.start = std::stod(csv_cell(cells, start_col));
        phone.end = std::stod(csv_cell(cells, end_col));
        phone.phone = csv_cell(cells, phone_col);
        phone.phone_base = strip_stress_digits(phone.phone);
        phone.word = csv_cell(cells, word_col);

        const std::string word_index = csv_cell(cells, word_index_col);
        const std::string speech_region_index = csv_cell(cells, speech_region_index_col);
        const std::string word_phone_index = csv_cell(cells, word_phone_index_col);
        const std::string sentence_index = csv_cell(cells, sentence_index_col);

        if (!word_index.empty()) phone.word_index = std::stoi(word_index);
        if (!speech_region_index.empty()) phone.speech_region_index = std::stoi(speech_region_index);
        if (!word_phone_index.empty()) phone.word_phone_index = std::stoi(word_phone_index);
        if (!sentence_index.empty()) phone.sentence_index = std::stoi(sentence_index);
        out.push_back(phone);
        ++global_index;
    }
    return out;
}

static std::vector<GoldWordTiming> read_gold_words_csv(const fs::path& path)
{
    std::vector<GoldWordTiming> out;
    std::ifstream file(path);
    if (!file) return out;

    std::string line;
    std::getline(file, line);
    const std::vector<std::string> header = parse_csv_cells(line);
    const bool has_header = !header.empty() && header[0] == "word_index";
    const int word_index_col = has_header ? header_index(header, "word_index") : 0;
    const int word_col = has_header ? header_index(header, "word") : 1;
    const int start_col = has_header ? header_index(header, "start") : 2;
    const int end_col = has_header ? header_index(header, "end") : 3;
    const int speech_region_index_col = has_header ? header_index(header, "speech_region_index") : 4;
    int sentence_index_col = has_header ? header_index(header, "text_sentence_index") : 5;
    if (sentence_index_col < 0 && has_header)
    {
        sentence_index_col = header_index(header, "sentence_index");
    }
    while (std::getline(file, line))
    {
        if (line.empty()) continue;
        const std::vector<std::string> cells = parse_csv_cells(line);
        if (word_col < 0 || start_col < 0 || end_col < 0) continue;
        GoldWordTiming word;
        const std::string word_index = csv_cell(cells, word_index_col);
        const std::string speech_region_index = csv_cell(cells, speech_region_index_col);
        const std::string sentence_index = csv_cell(cells, sentence_index_col);
        if (!word_index.empty()) word.word_index = std::stoi(word_index);
        word.word = csv_cell(cells, word_col);
        word.start = std::stod(csv_cell(cells, start_col));
        word.end = std::stod(csv_cell(cells, end_col));
        if (!speech_region_index.empty()) word.speech_region_index = std::stoi(speech_region_index);
        if (!sentence_index.empty()) word.sentence_index = std::stoi(sentence_index);
        out.push_back(word);
    }
    return out;
}

static std::vector<GoldSpeechRegion> read_gold_speech_csv(const fs::path& path)
{
    std::vector<GoldSpeechRegion> out;
    std::ifstream file(path);
    if (!file) return out;

    std::string line;
    std::getline(file, line);
    const std::vector<std::string> header = parse_csv_cells(line);
    const bool has_header = !header.empty() && header[0] == "index";
    const int index_col = has_header ? header_index(header, "index") : 0;
    const int start_col = has_header ? header_index(header, "start") : 1;
    const int end_col = has_header ? header_index(header, "end") : 2;
    int last_speech_col = has_header ? header_index(header, "last_speech") : 3;
    while (std::getline(file, line))
    {
        if (line.empty()) continue;
        const std::vector<std::string> cells = parse_csv_cells(line);
        if (start_col < 0 || end_col < 0) continue;
        GoldSpeechRegion region;
        const std::string index = csv_cell(cells, index_col);
        const std::string last_speech = csv_cell(cells, last_speech_col);
        if (!index.empty()) region.index = std::stoi(index);
        region.start = std::stod(csv_cell(cells, start_col));
        region.end = std::stod(csv_cell(cells, end_col));
        if (!last_speech.empty()) region.last_speech = std::stod(last_speech);
        out.push_back(region);
    }
    return out;
}

static std::vector<GoldBoundaryTiming> read_gold_boundaries_csv(const fs::path& path)
{
    std::vector<GoldBoundaryTiming> out;
    std::ifstream file(path);
    if (!file) return out;

    std::string line;
    std::getline(file, line);
    const std::vector<std::string> header = parse_csv_cells(line);
    const bool has_header = !header.empty() && header[0] == "word_index";
    const int word_index_col = has_header ? header_index(header, "word_index") : 0;
    const int word_col = has_header ? header_index(header, "word") : 1;
    const int next_word_index_col = has_header ? header_index(header, "next_word_index") : 2;
    const int next_word_col = has_header ? header_index(header, "next_word") : 3;
    const int mark_col = has_header ? header_index(header, "mark") : 4;
    const int pause_class_col = has_header ? header_index(header, "pause_class") : 6;
    const int split_applied_col = has_header ? header_index(header, "split_applied") : 7;
    const int direct_word_gap_seconds_col = has_header ? header_index(header, "direct_word_gap_seconds") : 9;
    const int phone_gap_seconds_col = has_header ? header_index(header, "phone_gap_seconds") : 10;
    const int wave_silence_seconds_col = has_header ? header_index(header, "wave_silence_seconds") : 11;
    const int acoustic_gap_seconds_col = has_header ? header_index(header, "acoustic_gap_seconds") : 12;
    const int speech_region_index_before_col = has_header ? header_index(header, "speech_region_index_before") : 13;
    const int speech_region_index_after_col = has_header ? header_index(header, "speech_region_index_after") : 14;

    while (std::getline(file, line))
    {
        if (line.empty()) continue;
        const std::vector<std::string> cells = parse_csv_cells(line);
        GoldBoundaryTiming boundary;

        const std::string word_index = csv_cell(cells, word_index_col);
        const std::string next_word_index = csv_cell(cells, next_word_index_col);
        const std::string split_applied = csv_cell(cells, split_applied_col);
        const std::string direct_word_gap_seconds = csv_cell(cells, direct_word_gap_seconds_col);
        const std::string phone_gap_seconds = csv_cell(cells, phone_gap_seconds_col);
        const std::string wave_silence_seconds = csv_cell(cells, wave_silence_seconds_col);
        const std::string acoustic_gap_seconds = csv_cell(cells, acoustic_gap_seconds_col);
        const std::string speech_region_index_before = csv_cell(cells, speech_region_index_before_col);
        const std::string speech_region_index_after = csv_cell(cells, speech_region_index_after_col);

        if (!word_index.empty()) boundary.word_index = std::stoi(word_index);
        boundary.word = csv_cell(cells, word_col);
        if (!next_word_index.empty()) boundary.next_word_index = std::stoi(next_word_index);
        boundary.next_word = csv_cell(cells, next_word_col);
        boundary.mark = csv_cell(cells, mark_col);
        boundary.pause_class = csv_cell(cells, pause_class_col);
        boundary.split_applied = !split_applied.empty() && split_applied != "0";
        if (!direct_word_gap_seconds.empty()) boundary.direct_word_gap_seconds = std::stod(direct_word_gap_seconds);
        if (!phone_gap_seconds.empty()) boundary.phone_gap_seconds = std::stod(phone_gap_seconds);
        if (!wave_silence_seconds.empty()) boundary.wave_silence_seconds = std::stod(wave_silence_seconds);
        if (!acoustic_gap_seconds.empty()) boundary.acoustic_gap_seconds = std::stod(acoustic_gap_seconds);
        if (!speech_region_index_before.empty()) boundary.speech_region_index_before = std::stoi(speech_region_index_before);
        if (!speech_region_index_after.empty()) boundary.speech_region_index_after = std::stoi(speech_region_index_after);
        out.push_back(std::move(boundary));
    }

    return out;
}

static std::string pause_class_name(EOffgridAIBoundaryPauseClass pause_class)
{
    switch (pause_class)
    {
    case EOffgridAIBoundaryPauseClass::HardBreakPause:
        return "hard_break_pause";
    case EOffgridAIBoundaryPauseClass::SoftListPause:
        return "soft_list_pause";
    case EOffgridAIBoundaryPauseClass::None:
    default:
        return "none";
    }
}

static std::string visual_phone_role_name(EOffgridAIVisualPhoneRole role)
{
    switch (role)
    {
    case EOffgridAIVisualPhoneRole::PrimaryPose:
        return "primary_pose";
    case EOffgridAIVisualPhoneRole::SupportingPose:
        return "supporting_pose";
    case EOffgridAIVisualPhoneRole::Coarticulated:
        return "coarticulated";
    case EOffgridAIVisualPhoneRole::TimingOnly:
    default:
        return "timing_only";
    }
}

static double landmark_detection_threshold(const std::string& type)
{
    if (type == "mbp") return 0.55;
    if (type == "fv") return 0.50;
    if (type == "w") return 0.45;
    if (type == "chjjsh") return 0.50;
    if (type == "round") return 0.55;
    if (is_pause_lull_type(type)) return 0.62;
    return 1.0;
}

static double landmark_reference_duration_seconds(const std::string& type)
{
    // MFA-derived family durations from the checked-in corpus:
    // mbp median~70ms p75~100ms
    // fv median~90ms p75~110ms
    // w median~80ms p75~120ms
    // chjjsh median~110ms p75~140ms
    // round median~110ms p75~160ms
    // comma_lull is a boundary event rather than a phone family; keep a wider window.
    if (type == "mbp") return 0.070;
    if (type == "fv") return 0.090;
    if (type == "w") return 0.080;
    if (type == "chjjsh") return 0.110;
    if (type == "round") return 0.110;
    if (is_pause_lull_type(type)) return 0.200;
    return 0.100;
}

static double landmark_match_tolerance_seconds(const std::string& type)
{
    (void)type;
    return 0.100;
}

static double landmark_conditioned_half_window_seconds(const std::string& type)
{
    // Text durations are useful pacing priors, not absolute alignment. Search
    // enough streamed context for the monotonic sequence model to recover from
    // accumulated prior drift (roughly 0.55 s on this corpus).
    if (type == "mbp") return 0.750;
    if (type == "fv") return 0.750;
    if (type == "w") return 0.750;
    if (type == "chjjsh") return 0.750;
    if (type == "round") return 0.750;
    if (is_pause_lull_type(type)) return 0.750;
    return 0.100;
}

static double landmark_conditioned_detection_threshold(const std::string& type)
{
    if (type == "mbp") return 0.27;
    if (type == "fv") return 0.28;
    if (type == "w") return 0.24;
    if (type == "chjjsh") return 0.32;
    if (type == "round") return 0.32;
    if (is_pause_lull_type(type)) return 0.44;
    return landmark_detection_threshold(type);
}

static double landmark_conditioned_distance_penalty_scale(const std::string& type)
{
    if (type == "mbp") return 0.12;
    if (type == "fv") return 0.14;
    if (type == "w") return 0.28;
    if (type == "chjjsh") return 0.24;
    if (type == "round") return 0.32;
    if (is_pause_lull_type(type)) return 0.14;
    return 0.18;
}

static double landmark_score_for_type(const std::string& type, const FOffgridAIArticulatoryProbabilityField& field)
{
    if (type == "mbp")
    {
        return std::max({
            static_cast<double>(field.PhoneScores.Bilabial),
            static_cast<double>(field.Closure * 0.70f),
            static_cast<double>(field.Release * 0.35f)
        });
    }
    if (type == "fv")
    {
        return std::max(
            static_cast<double>(field.PhoneScores.Labiodental),
            static_cast<double>(field.Fricative * 0.85f));
    }
    if (type == "w")
    {
        return std::max({
            static_cast<double>(field.PhoneScores.Glide),
            static_cast<double>(field.PhoneScores.Liquid * 0.50f),
            static_cast<double>(field.PhoneScores.VowelRound * 0.85f),
            static_cast<double>(field.Sonorant * 0.60f)
        });
    }
    if (type == "chjjsh")
    {
        return std::max(
            static_cast<double>(field.PhoneScores.Sibilant),
            static_cast<double>(field.PhoneScores.StopBurst * 0.55f));
    }
    if (type == "round")
    {
        return std::max(
            static_cast<double>(field.PhoneScores.VowelRound),
            static_cast<double>(field.Vowel * 0.50f));
    }
    return 0.0;
}

static double clamp01(double x)
{
    return std::max(0.0, std::min(1.0, x));
}

static double comma_lull_score_for_frame(const FOffgridAIStreamingAudioFeatureFrame& frame)
{
    const std::string pause_family = derived_pause_family(frame);
    const double pause_confidence = derived_pause_confidence(frame);
    const bool pause_family_support =
        pause_family != "none" &&
        pause_family != "continuous_speech" &&
        pause_confidence >= 0.35;
    const bool lull_gate =
        frame.bLocalRMSValley ||
        frame.bStrongQuiet ||
        static_cast<double>(frame.SilenceAccumSec) >= 0.030 ||
        pause_family_support;
    if (!lull_gate)
    {
        return 0.0;
    }

    const double low_rms = clamp01((0.12 - static_cast<double>(frame.RMSNorm)) / 0.12);
    const double low_evidence = clamp01((0.24 - static_cast<double>(frame.SpeechEvidence)) / 0.24);
    const double silence_accum = clamp01(static_cast<double>(frame.SilenceAccumSec) / 0.080);
    const double pause_conf = (pause_family != "none" && pause_family != "continuous_speech")
        ? pause_confidence
        : 0.0;
    const double valley = frame.bLocalRMSValley ? 1.0 : 0.0;
    const double strong_quiet = frame.bStrongQuiet ? 1.0 : 0.0;
    return clamp01(
        low_rms * 0.28 +
        low_evidence * 0.28 +
        silence_accum * 0.16 +
        pause_conf * 0.14 +
        valley * 0.06 +
        strong_quiet * 0.08);
}

static double landmark_score_for_frame(
    const std::string& type,
    const FOffgridAIStreamingAudioFeatureFrame& frame)
{
    if (is_pause_lull_type(type))
    {
        return comma_lull_score_for_frame(frame);
    }
    return landmark_score_for_type(type, FOffgridAIAcousticEvidence::BuildArticulatoryProbabilityField(frame));
}



static void build_runtime_phone_prior_times(
    const FOffgridAITextVisemePlan& plan,
    std::vector<double>& out_start_by_index,
    std::vector<double>& out_end_by_index,
    std::vector<double>& out_word_start_by_index,
    std::vector<double>& out_word_end_by_index)
{
    constexpr double kActiveDurationScale = 0.90;
    constexpr double kMinPhoneSeconds = 0.018;
    constexpr double kInterWordSpacerSeconds = 0.020;

    out_start_by_index.assign(plan.ExpectedPhones.Num(), 0.0);
    out_end_by_index.assign(plan.ExpectedPhones.Num(), 0.0);
    out_word_start_by_index.assign(plan.WordPhoneBeginIndices.Num(), 0.0);
    out_word_end_by_index.assign(plan.WordPhoneBeginIndices.Num(), 0.0);
    if (plan.ExpectedPhones.Num() <= 0)
    {
        return;
    }

    double cursor = 0.0;
    for (int32 phone_index = 0; phone_index < plan.ExpectedPhones.Num(); ++phone_index)
    {
        const FOffgridAIExpectedPhone& phone = plan.ExpectedPhones[phone_index];
        const int32 word_index = phone.WordIndex;
        const double duration = std::max(
            static_cast<double>(phone.WeightSeconds) * kActiveDurationScale,
            kMinPhoneSeconds);

        if (word_index >= 0
            && word_index < static_cast<int32>(out_word_start_by_index.size())
            && phone_index == plan.WordPhoneBeginIndices[word_index])
        {
            out_word_start_by_index[static_cast<size_t>(word_index)] = cursor;
        }

        out_start_by_index[static_cast<size_t>(phone_index)] = cursor;
        cursor += duration;
        out_end_by_index[static_cast<size_t>(phone_index)] = cursor;

        const bool is_last_phone_of_word =
            word_index >= 0
            && word_index < plan.WordPhoneEndIndices.Num()
            && (phone_index + 1) == plan.WordPhoneEndIndices[word_index];
        if (!is_last_phone_of_word)
        {
            continue;
        }

        if (word_index >= 0 && word_index < static_cast<int32>(out_word_end_by_index.size()))
        {
            out_word_end_by_index[static_cast<size_t>(word_index)] = cursor;
        }

        if ((phone_index + 1) < plan.ExpectedPhones.Num())
        {
            double boundary_pause_seconds = kInterWordSpacerSeconds;
            if (word_index >= 0 && plan.WordBoundaryPauseSecondsAfter.IsValidIndex(word_index))
            {
                boundary_pause_seconds = std::max(
                    boundary_pause_seconds,
                    static_cast<double>(plan.WordBoundaryPauseSecondsAfter[word_index]));
            }
            cursor += boundary_pause_seconds;
        }
    }
}

static std::string prosodic_vowel_family(const std::string& phone)
{
    if (phone == "AA" || phone == "AE" || phone == "AH"
        || phone == "AW" || phone == "AY")
        return "open";
    if (phone == "AO" || phone == "OW" || phone == "OY"
        || phone == "UH" || phone == "UW")
        return "round";
    return "front";
}


static std::vector<ProsodicPeakTarget> build_prosodic_peak_targets(
    const FOffgridAITextVisemePlan& plan,
    const std::vector<GoldPhoneTiming>& gold_phones)
{
    std::map<int, const GoldPhoneTiming*> gold_by_index;
    for (const GoldPhoneTiming& phone : gold_phones)
    {
        gold_by_index[phone.global_phone_index] = &phone;
    }

    std::vector<double> phone_starts;
    std::vector<double> phone_ends;
    std::vector<double> word_starts;
    std::vector<double> word_ends;
    build_runtime_phone_prior_times(plan, phone_starts, phone_ends, word_starts, word_ends);

    std::vector<ProsodicPeakTarget> targets;
    for (int32 i = 0; i < plan.ExpectedPhones.Num(); ++i)
    {
        const FOffgridAIExpectedPhone& expected = plan.ExpectedPhones[i];
        if (!expected.bIsVowel)
        {
            continue;
        }
        const std::string phone = to_std(expected.Phone);
        int stress = 0;
        if (!phone.empty() && std::isdigit(static_cast<unsigned char>(phone.back())) != 0)
        {
            stress = phone.back() - '0';
        }
        ProsodicPeakTarget target;
        target.target_index = static_cast<int>(targets.size());
        target.expected_phone_index = i;
        target.global_phone_index = expected.PhoneIndex;
        target.word_index = expected.WordIndex;
        target.speech_region_index = expected.SpeechRegionIndex;
        target.stress = stress;
        target.word = to_std(expected.SourceWord);
        target.phone = phone;
        target.vowel_family = prosodic_vowel_family(to_std(expected.BasePhone));
        target.prior_center = 0.5 * (phone_starts[static_cast<size_t>(i)] + phone_ends[static_cast<size_t>(i)]);
        const auto gold_it = gold_by_index.find(expected.PhoneIndex);
        if (gold_it != gold_by_index.end())
        {
            target.gold_start = gold_it->second->start;
            target.gold_end = gold_it->second->end;
            target.gold_center = 0.5 * (gold_it->second->start + gold_it->second->end);
            target.has_gold = true;
        }
        targets.push_back(std::move(target));
    }
    return targets;
}


static ProsodicPeakReport grade_prosodic_peaks(
    const std::vector<ProsodicPeakTarget>& targets,
    std::vector<ProsodicPeakObservation>& observations,
    const std::vector<ProsodicPeakObservation>& raw_peaks)
{
    ProsodicPeakReport report;
    report.available = true;
    report.target_count = static_cast<int>(std::count_if(targets.begin(), targets.end(), [](const auto& target) { return target.has_gold; }));
    report.observation_count = static_cast<int>(observations.size());
    std::vector<double> errors;
    std::vector<double> latencies;
    for (ProsodicPeakObservation& observation : observations)
    {
        if (observation.target_index < 0 || observation.target_index >= static_cast<int>(targets.size())) continue;
        const ProsodicPeakTarget& target = targets[static_cast<size_t>(observation.target_index)];
        if (!target.has_gold) continue;
        observation.error_ms = std::abs(observation.peak_sec - target.gold_center) * 1000.0;
        if (observation.error_ms <= 100.0)
        {
            observation.matched = true;
            ++report.matched_count;
            errors.push_back(observation.error_ms);
            latencies.push_back(std::max(0.0, observation.decision_sec - observation.peak_sec) * 1000.0);
        }
    }
    report.precision = report.observation_count > 0 ? static_cast<double>(report.matched_count) / report.observation_count : 0.0;
    report.recall = report.target_count > 0 ? static_cast<double>(report.matched_count) / report.target_count : 0.0;
    report.mean_abs_error_ms = mean_ms(errors);
    report.median_abs_error_ms = percentile_ms(errors, 0.50);
    report.p90_abs_error_ms = percentile_ms(errors, 0.90);
    report.mean_decision_latency_ms = mean_ms(latencies);
    report.matched_error_samples_ms = errors;

    std::vector<std::pair<double, double>> gold_intervals;
    for (const ProsodicPeakTarget& target : targets)
    {
        if (target.has_gold) gold_intervals.emplace_back(target.gold_start, target.gold_end);
    }
    std::vector<double> raw_errors;
    report.raw_observation_count = static_cast<int>(raw_peaks.size());

    // Both streams are chronological.  A nearest-unused greedy match can
    // consume a later vowel for an early peak and manufacture one FP plus one
    // FN in dense speech.  Find the monotonic assignment with the most matches,
    // breaking ties by total distance outside the MFA interval.
    struct MatchScore
    {
        int count = 0;
        double error = 0.0;
    };
    const size_t peak_count = raw_peaks.size();
    const size_t target_count = gold_intervals.size();
    std::vector<std::vector<MatchScore>> dp(
        peak_count + 1, std::vector<MatchScore>(target_count + 1));
    std::vector<std::vector<char>> choice(
        peak_count + 1, std::vector<char>(target_count + 1, 'p'));
    auto Better = [](const MatchScore& A, const MatchScore& B)
    {
        return A.count > B.count || (A.count == B.count && A.error < B.error);
    };
    for (size_t i = 1; i <= peak_count; ++i)
    {
        for (size_t j = 1; j <= target_count; ++j)
        {
            MatchScore best = dp[i - 1][j];
            char best_choice = 'p';
            if (Better(dp[i][j - 1], best))
            {
                best = dp[i][j - 1];
                best_choice = 't';
            }
            const auto [start, end] = gold_intervals[j - 1];
            const double error = raw_peaks[i - 1].peak_sec < start
                ? start - raw_peaks[i - 1].peak_sec
                : (raw_peaks[i - 1].peak_sec > end ? raw_peaks[i - 1].peak_sec - end : 0.0);
            MatchScore matched = dp[i - 1][j - 1];
            ++matched.count;
            matched.error += error;
            if (error <= 0.100 && Better(matched, best))
            {
                best = matched;
                best_choice = 'm';
            }
            dp[i][j] = best;
            choice[i][j] = best_choice;
        }
    }
    size_t i = peak_count;
    size_t j = target_count;
    while (i > 0 && j > 0)
    {
        if (choice[i][j] == 'm')
        {
            const auto [start, end] = gold_intervals[j - 1];
            const double peak = raw_peaks[i - 1].peak_sec;
            raw_errors.push_back((peak < start ? start - peak : (peak > end ? peak - end : 0.0)) * 1000.0);
            --i;
            --j;
        }
        else if (choice[i][j] == 't')
        {
            --j;
        }
        else
        {
            --i;
        }
    }
    report.raw_matched_count = dp[peak_count][target_count].count;
    report.raw_precision = report.raw_observation_count > 0
        ? static_cast<double>(report.raw_matched_count) / report.raw_observation_count : 0.0;
    report.raw_recall = report.target_count > 0
        ? static_cast<double>(report.raw_matched_count) / report.target_count : 0.0;
    report.raw_mean_abs_error_ms = mean_ms(raw_errors);
    return report;
}

// Transcript and MFA phone inventories can differ through reductions, inserted
// SPN intervals, or pronunciation variants. Neural syllable scoring therefore
// aligns the complete chronological vowel streams instead of joining them by
// raw phone index. Every MFA vowel remains a target and every neural vowel
// remains an observation; only pairs within the existing 100 ms criterion can
// match.
static ProsodicPeakReport grade_monotonic_neural_syllables(
    const std::vector<ProsodicPeakTarget>& targets,
    const std::vector<ProsodicPeakObservation>& observations)
{
    std::vector<const ProsodicPeakTarget*> gold;
    for (const ProsodicPeakTarget& target : targets)
        if (target.has_gold) gold.push_back(&target);

    struct MatchScore
    {
        int Count = 0;
        double ErrorMs = 0.0;
    };
    const size_t observation_count = observations.size();
    const size_t target_count = gold.size();
    std::vector<std::vector<MatchScore>> dp(
        observation_count + 1, std::vector<MatchScore>(target_count + 1));
    std::vector<std::vector<char>> choice(
        observation_count + 1, std::vector<char>(target_count + 1, 'o'));
    auto better = [](const MatchScore& left, const MatchScore& right) {
        return left.Count > right.Count
            || (left.Count == right.Count && left.ErrorMs < right.ErrorMs);
    };
    for (size_t observation = 1; observation <= observation_count; ++observation)
    {
        for (size_t target = 1; target <= target_count; ++target)
        {
            MatchScore best = dp[observation - 1][target];
            char step = 'o';
            if (better(dp[observation][target - 1], best))
            {
                best = dp[observation][target - 1];
                step = 't';
            }
            const double error_ms = std::abs(
                observations[observation - 1].peak_sec - gold[target - 1]->gold_center) * 1000.0;
            if (error_ms <= 100.0)
            {
                MatchScore matched = dp[observation - 1][target - 1];
                ++matched.Count;
                matched.ErrorMs += error_ms;
                if (better(matched, best))
                {
                    best = matched;
                    step = 'm';
                }
            }
            dp[observation][target] = best;
            choice[observation][target] = step;
        }
    }

    std::vector<double> errors;
    std::vector<double> latencies;
    size_t observation = observation_count;
    size_t target = target_count;
    while (observation > 0 && target > 0)
    {
        const char step = choice[observation][target];
        if (step == 'm')
        {
            const auto& item = observations[observation - 1];
            errors.push_back(std::abs(item.peak_sec - gold[target - 1]->gold_center) * 1000.0);
            latencies.push_back(std::max(0.0, item.decision_sec - item.peak_sec) * 1000.0);
            --observation;
            --target;
        }
        else if (step == 't') --target;
        else --observation;
    }

    ProsodicPeakReport report;
    report.available = true;
    report.target_count = static_cast<int>(target_count);
    report.observation_count = static_cast<int>(observation_count);
    report.matched_count = dp[observation_count][target_count].Count;
    report.precision = observation_count > 0
        ? static_cast<double>(report.matched_count) / observation_count : 0.0;
    report.recall = target_count > 0
        ? static_cast<double>(report.matched_count) / target_count : 0.0;
    report.mean_abs_error_ms = mean_ms(errors);
    report.median_abs_error_ms = percentile_ms(errors, 0.50);
    report.p90_abs_error_ms = percentile_ms(errors, 0.90);
    report.mean_decision_latency_ms = mean_ms(latencies);
    report.matched_error_samples_ms = std::move(errors);
    return report;
}

static std::vector<ProsodicPeakObservation> neural_syllable_observations(
    const std::vector<ProsodicPeakTarget>& targets,
    const FOffgridAIAlignedVisemeTrack& track)
{
    std::vector<ProsodicPeakObservation> observations;
    for (const ProsodicPeakTarget& target : targets)
    {
        const auto event = std::find_if(track.Events.begin(), track.Events.end(), [&](const auto& candidate) {
            return candidate.bIsRenderable
                && !candidate.bCanceledByWordHandoff
                && candidate.SourcePhoneIndex == target.global_phone_index;
        });
        if (event == track.Events.end()) continue;
        ProsodicPeakObservation observation;
        observation.target_index = target.target_index;
        observation.peak_sec = event->FinalRenderCenterSeconds;
        observation.decision_sec = event->FinalRenderCenterSeconds + 0.350;
        observation.prominence = event->Strength;
        observation.family_score = 1.0;
        observation.score = 1.0;
        observation.speech_region_index = event->SpeechRegionIndex;
        observations.push_back(observation);
    }
    return observations;
}

static bool is_gold_vowel_phone(const std::string& phone_base)
{
    static const std::vector<std::string> vowels = {
        "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER",
        "EY", "IH", "IY", "OW", "OY", "UH", "UW"
    };
    return std::find(vowels.begin(), vowels.end(), phone_base) != vowels.end();
}

// Audio-only nucleus grading must use MFA's vowel intervals directly. Mapping
// transcript vowel ordinals onto the MFA phone sequence is appropriate for
// transcript-assignment diagnostics, but pronunciation disagreements can put
// those ordinals on consonants and corrupt an acoustic detector score.
static std::vector<ProsodicPeakTarget> build_mfa_nucleus_targets(
    const std::vector<GoldPhoneTiming>& gold_phones)
{
    std::vector<ProsodicPeakTarget> targets;
    for (const GoldPhoneTiming& phone : gold_phones)
    {
        if (!is_gold_vowel_phone(phone.phone_base)) continue;
        ProsodicPeakTarget target;
        target.target_index = static_cast<int>(targets.size());
        target.expected_phone_index = phone.global_phone_index;
        target.global_phone_index = phone.global_phone_index;
        target.word_index = phone.word_index;
        target.speech_region_index = phone.speech_region_index;
        target.word = phone.word;
        target.phone = phone.phone;
        target.vowel_family = prosodic_vowel_family(phone.phone_base);
        if (!phone.phone.empty()
            && std::isdigit(static_cast<unsigned char>(phone.phone.back())) != 0)
        {
            target.stress = phone.phone.back() - '0';
        }
        target.gold_start = phone.start;
        target.gold_end = phone.end;
        target.gold_center = 0.5 * (phone.start + phone.end);
        target.prior_center = target.gold_center;
        target.has_gold = true;
        targets.push_back(std::move(target));
    }
    return targets;
}

static std::vector<EvidenceIngredientTarget> build_evidence_ingredient_targets(
    const std::vector<GoldPhoneTiming>& gold_phones,
    const std::vector<GoldWordTiming>& gold_words,
    const std::vector<GoldSpeechRegion>& gold_speech,
    const std::vector<GoldBoundaryTiming>& gold_boundaries)
{
    std::vector<EvidenceIngredientTarget> targets;
    for (const GoldPhoneTiming& phone : gold_phones)
    {
        if (is_gold_vowel_phone(phone.phone_base))
        {
            targets.push_back({ "pulse", phone.start, phone.end });
            if (phone.phone_base == "AA" || phone.phone_base == "AE" || phone.phone_base == "AH" ||
                phone.phone_base == "AW" || phone.phone_base == "AY")
                targets.push_back({ "open", phone.start, phone.end });
            else if (phone.phone_base == "EH" || phone.phone_base == "ER" || phone.phone_base == "EY" ||
                phone.phone_base == "IH" || phone.phone_base == "IY")
                targets.push_back({ "front", phone.start, phone.end });
        }
        if (phone.phone_base == "M" || phone.phone_base == "B" || phone.phone_base == "P")
            targets.push_back({ "mbp", phone.start, phone.end });
        else if (phone.phone_base == "F" || phone.phone_base == "V")
            targets.push_back({ "fv", phone.start, phone.end });
        else if (phone.phone_base == "W" || phone.phone_base == "Y")
            targets.push_back({ "glide", phone.start, phone.end });
        else if (phone.phone_base == "S" || phone.phone_base == "Z" || phone.phone_base == "SH" ||
            phone.phone_base == "ZH" || phone.phone_base == "CH" || phone.phone_base == "JH")
            targets.push_back({ "sibilant", phone.start, phone.end });
        else if (phone.phone_base == "UW" || phone.phone_base == "UH" || phone.phone_base == "OW" ||
            phone.phone_base == "OY" || phone.phone_base == "AO")
            targets.push_back({ "round", phone.start, phone.end });
    }
    for (const auto& region : gold_speech)
    {
        targets.push_back({ "lull", region.end, region.end });
        targets.push_back({ "resume", region.start, region.start });
    }
    std::map<int, const GoldWordTiming*> words_by_index;
    for (const auto& word : gold_words) words_by_index[word.word_index] = &word;
    for (const auto& boundary : gold_boundaries)
    {
        const double gap = std::max({
            boundary.direct_word_gap_seconds,
            boundary.phone_gap_seconds,
            boundary.acoustic_gap_seconds
        });
        const auto previous = words_by_index.find(boundary.word_index);
        const auto next = words_by_index.find(boundary.next_word_index);
        if (gap < 0.030 || previous == words_by_index.end() || next == words_by_index.end()) continue;
        targets.push_back({ "lull", previous->second->end, previous->second->end });
        targets.push_back({ "resume", next->second->start, next->second->start });
    }
    std::sort(targets.begin(), targets.end(), [](const auto& a, const auto& b) {
        if (a.type != b.type) return a.type < b.type;
        return a.start < b.start;
    });
    targets.erase(std::unique(targets.begin(), targets.end(), [](const auto& a, const auto& b) {
        return a.type == b.type && std::abs(a.start - b.start) <= 0.030;
    }), targets.end());
    return targets;
}

static EvidenceIngredientReport grade_evidence_ingredients(
    const std::vector<EvidenceIngredientTarget>& targets,
    const TArray<FOffgridAIAudioLandmarkObservation>& observations,
    const std::vector<GoldSpeechRegion>& gold_speech,
    double preroll_ms,
    double postroll_ms)
{
    EvidenceIngredientReport report;
    report.available = true;
    report.preroll_ms = preroll_ms;
    report.postroll_ms = postroll_ms;

    std::map<std::string, std::vector<EvidenceIngredientTarget>> targets_by_type;
    std::map<std::string, std::vector<const FOffgridAIAudioLandmarkObservation*>> observations_by_type;
    for (const auto& target : targets) targets_by_type[target.type].push_back(target);
    for (const auto& observation : observations)
    {
        observations_by_type[to_std(FOffgridAIStreamingEvidenceSurface::LandmarkTypeToString(observation.Type))]
            .push_back(&observation);
    }

    for (const auto& [type, type_targets] : targets_by_type)
    {
        EvidenceIngredientTypeReport type_report;
        type_report.target_count = static_cast<int>(type_targets.size());
        const auto observation_it = observations_by_type.find(type);
        const std::vector<const FOffgridAIAudioLandmarkObservation*> type_observations =
            observation_it != observations_by_type.end()
            ? observation_it->second
            : std::vector<const FOffgridAIAudioLandmarkObservation*>();
        type_report.observation_count = static_cast<int>(type_observations.size());

        std::vector<bool> used_targets(type_targets.size(), false);
        std::vector<double> errors;
        std::vector<double> latencies;
        for (const auto* observation : type_observations)
        {
            int best_target = -1;
            double best_error = 0.100001;
            for (size_t target_index = 0; target_index < type_targets.size(); ++target_index)
            {
                if (used_targets[target_index]) continue;
                const auto& target = type_targets[target_index];
                const double error = observation->CenterSec < target.start
                    ? target.start - observation->CenterSec
                    : (observation->CenterSec > target.end ? observation->CenterSec - target.end : 0.0);
                if (error < best_error)
                {
                    best_error = error;
                    best_target = static_cast<int>(target_index);
                }
            }
            if (best_target >= 0)
            {
                used_targets[static_cast<size_t>(best_target)] = true;
                ++type_report.matched_count;
                errors.push_back(best_error * 1000.0);
                latencies.push_back(std::max(0.0, static_cast<double>(observation->DecisionSec - observation->CenterSec)) * 1000.0);
            }
        }
        type_report.precision = type_report.observation_count > 0
            ? static_cast<double>(type_report.matched_count) / type_report.observation_count : 0.0;
        type_report.recall = type_report.target_count > 0
            ? static_cast<double>(type_report.matched_count) / type_report.target_count : 0.0;
        type_report.mean_abs_error_ms = mean_ms(errors);
        type_report.mean_decision_latency_ms = mean_ms(latencies);

        if (type == "pulse")
        {
            auto target_center = [](const EvidenceIngredientTarget& target) {
                return 0.5 * (target.start + target.end);
            };
            for (const GoldSpeechRegion& region : gold_speech)
            {
                const int target_count = static_cast<int>(std::count_if(
                    type_targets.begin(), type_targets.end(), [&](const auto& target) {
                        const double center = target_center(target);
                        return center >= region.start && center <= region.end;
                    }));
                const int observed_count = static_cast<int>(std::count_if(
                    type_observations.begin(), type_observations.end(), [&](const auto* observation) {
                        return observation->CenterSec >= region.start && observation->CenterSec <= region.end;
                    }));
                ++type_report.region_count;
                if (target_count == observed_count) ++type_report.exact_region_count;
                type_report.region_count_abs_error += std::abs(target_count - observed_count);
            }

            auto accumulate_clustered = [&](double maximum_neighbor_gap,
                                            int& clustered_targets,
                                            int& clustered_matches) {
                for (size_t target_index = 0; target_index < type_targets.size(); ++target_index)
                {
                    const double center = target_center(type_targets[target_index]);
                    bool clustered = false;
                    if (target_index > 0)
                        clustered = center - target_center(type_targets[target_index - 1]) <= maximum_neighbor_gap;
                    if (target_index + 1 < type_targets.size())
                        clustered = clustered ||
                            target_center(type_targets[target_index + 1]) - center <= maximum_neighbor_gap;
                    if (!clustered) continue;
                    ++clustered_targets;
                    if (used_targets[target_index]) ++clustered_matches;
                }
            };
            accumulate_clustered(0.120, type_report.clustered_120_target_count,
                type_report.clustered_120_matched_count);
            accumulate_clustered(0.150, type_report.clustered_150_target_count,
                type_report.clustered_150_matched_count);
            accumulate_clustered(0.200, type_report.clustered_200_target_count,
                type_report.clustered_200_matched_count);

            for (const auto* observation : type_observations)
            {
                const bool in_speech = std::any_of(gold_speech.begin(), gold_speech.end(),
                    [&](const GoldSpeechRegion& region) {
                        return observation->CenterSec >= region.start - 0.030 &&
                            observation->CenterSec <= region.end + 0.030;
                    });
                if (!in_speech) ++type_report.outside_speech_observation_count;
            }
        }
        report.by_type[type] = type_report;
    }
    return report;
}

static std::string phone_family_for_intra_envelope_grade(const std::string& base)
{
    if (base == "M" || base == "B" || base == "P") return "mbp";
    if (base == "F" || base == "V") return "fv";
    if (base == "W" || base == "Y") return "glide";
    if (base == "S" || base == "Z" || base == "SH" || base == "ZH"
        || base == "CH" || base == "JH") return "sibilant";
    if (base == "UW" || base == "UH" || base == "OW" || base == "OY" || base == "AO") return "round";
    return {};
}

static EvidenceIngredientReport grade_intra_envelope_phone_evidence(
    const FOffgridAITextVisemePlan& plan,
    const std::vector<GoldPhoneTiming>& gold_phones,
    const TArray<FOffgridAIAudioLandmarkObservation>& observations,
    double preroll_ms,
    double postroll_ms)
{
    struct SyllableGold
    {
        double start = 0.0;
        double end = 0.0;
        double nucleus_start = 0.0;
        double nucleus_end = 0.0;
        std::set<std::string> expected_families;
        std::map<std::string, std::vector<EvidenceIngredientTarget>> families;
    };

    std::map<int, const GoldPhoneTiming*> gold_by_index;
    for (const GoldPhoneTiming& phone : gold_phones) gold_by_index[phone.global_phone_index] = &phone;

    std::vector<SyllableGold> syllables;
    for (const FOffgridAIPlannedSyllable& syllable : plan.Syllables)
    {
        if (!plan.ExpectedPhones.IsValidIndex(syllable.NucleusPhoneIndex)) continue;
        const auto nucleus_it = gold_by_index.find(plan.ExpectedPhones[syllable.NucleusPhoneIndex].PhoneIndex);
        if (nucleus_it == gold_by_index.end()) continue;

        SyllableGold gold;
        gold.start = nucleus_it->second->start;
        gold.end = nucleus_it->second->end;
        gold.nucleus_start = nucleus_it->second->start;
        gold.nucleus_end = nucleus_it->second->end;
        for (int32 phone_index = syllable.PhoneBeginIndex;
             phone_index < syllable.PhoneEndIndex && plan.ExpectedPhones.IsValidIndex(phone_index);
             ++phone_index)
        {
            const std::string expected_family = phone_family_for_intra_envelope_grade(
                to_std(plan.ExpectedPhones[phone_index].BasePhone));
            if (!expected_family.empty()) gold.expected_families.insert(expected_family);
            const auto phone_it = gold_by_index.find(plan.ExpectedPhones[phone_index].PhoneIndex);
            if (phone_it == gold_by_index.end()) continue;
            gold.start = FMath::Min(gold.start, phone_it->second->start);
            gold.end = FMath::Max(gold.end, phone_it->second->end);
            const std::string family = phone_family_for_intra_envelope_grade(phone_it->second->phone_base);
            if (!family.empty())
                gold.families[family].push_back({family, phone_it->second->start, phone_it->second->end});
        }
        syllables.push_back(std::move(gold));
    }

    std::vector<const FOffgridAIAudioLandmarkObservation*> pulses;
    for (const auto& observation : observations)
        if (observation.Type == EOffgridAIAudioLandmarkType::SyllabicPulse) pulses.push_back(&observation);

    std::vector<int> pulse_for_syllable(syllables.size(), -1);
    std::vector<bool> used_syllables(syllables.size(), false);
    for (size_t pulse_index = 0; pulse_index < pulses.size(); ++pulse_index)
    {
        int best = -1;
        double best_error = 0.100001;
        for (size_t syllable_index = 0; syllable_index < syllables.size(); ++syllable_index)
        {
            if (used_syllables[syllable_index]) continue;
            const SyllableGold& syllable = syllables[syllable_index];
            const double error = pulses[pulse_index]->CenterSec < syllable.nucleus_start
                ? syllable.nucleus_start - pulses[pulse_index]->CenterSec
                : (pulses[pulse_index]->CenterSec > syllable.nucleus_end
                    ? pulses[pulse_index]->CenterSec - syllable.nucleus_end : 0.0);
            if (error < best_error)
            {
                best = static_cast<int>(syllable_index);
                best_error = error;
            }
        }
        if (best >= 0)
        {
            used_syllables[static_cast<size_t>(best)] = true;
            pulse_for_syllable[static_cast<size_t>(best)] = static_cast<int>(pulse_index);
        }
    }

    EvidenceIngredientReport report;
    report.available = true;
    report.preroll_ms = preroll_ms;
    report.postroll_ms = postroll_ms;
    const std::vector<std::string> families = {"mbp", "fv", "glide", "sibilant", "round"};
    for (const std::string& family : families) report.by_type[family] = EvidenceIngredientTypeReport{};

    for (size_t syllable_index = 0; syllable_index < syllables.size(); ++syllable_index)
    {
        if (pulse_for_syllable[syllable_index] < 0) continue;
        const SyllableGold& syllable = syllables[syllable_index];
        for (const std::string& family : families)
        {
            // Transcript identity owns which detector is relevant. The acoustic
            // channel only confirms and locates that expected family inside the
            // already pulse-matched syllable; it never invents phone identity.
            if (syllable.expected_families.find(family) == syllable.expected_families.end()) continue;
            EvidenceIngredientTypeReport& stats = report.by_type[family];
            const auto target_it = syllable.families.find(family);
            const std::vector<EvidenceIngredientTarget> targets = target_it != syllable.families.end()
                ? target_it->second : std::vector<EvidenceIngredientTarget>{};
            std::vector<const FOffgridAIAudioLandmarkObservation*> candidates;
            for (const auto& observation : observations)
            {
                if (to_std(FOffgridAIStreamingEvidenceSurface::LandmarkTypeToString(observation.Type)) != family) continue;
                if (observation.CenterSec >= syllable.start && observation.CenterSec <= syllable.end)
                    candidates.push_back(&observation);
            }
            stats.target_count += static_cast<int>(targets.size());
            stats.observation_count += static_cast<int>(candidates.size());
            std::vector<bool> used_targets(targets.size(), false);
            for (const auto* candidate : candidates)
            {
                int best = -1;
                double best_error = 1.0e9;
                for (size_t target_index = 0; target_index < targets.size(); ++target_index)
                {
                    if (used_targets[target_index]) continue;
                    const double error = candidate->CenterSec < targets[target_index].start
                        ? targets[target_index].start - candidate->CenterSec
                        : (candidate->CenterSec > targets[target_index].end
                            ? candidate->CenterSec - targets[target_index].end : 0.0);
                    if (error < best_error)
                    {
                        best = static_cast<int>(target_index);
                        best_error = error;
                    }
                }
                if (best < 0) continue;
                used_targets[static_cast<size_t>(best)] = true;
                ++stats.matched_count;
                stats.mean_abs_error_ms += best_error * 1000.0;
                stats.mean_decision_latency_ms +=
                    FMath::Max(static_cast<double>(candidate->DecisionSec - candidate->CenterSec), 0.0) * 1000.0;
            }
        }
    }
    for (auto& [family, stats] : report.by_type)
    {
        stats.precision = stats.observation_count > 0
            ? static_cast<double>(stats.matched_count) / stats.observation_count : 0.0;
        stats.recall = stats.target_count > 0
            ? static_cast<double>(stats.matched_count) / stats.target_count : 0.0;
        if (stats.matched_count > 0)
        {
            stats.mean_abs_error_ms /= stats.matched_count;
            stats.mean_decision_latency_ms /= stats.matched_count;
        }
    }
    return report;
}

static std::string evidence_ingredient_observations_csv(
    const TArray<FOffgridAIAudioLandmarkObservation>& observations)
{
    std::ostringstream out;
    out << "type,start,end,center,decision,score\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& observation : observations)
    {
        out << to_std(FOffgridAIStreamingEvidenceSurface::LandmarkTypeToString(observation.Type)) << ','
            << observation.StartSec << ',' << observation.EndSec << ',' << observation.CenterSec << ','
            << observation.DecisionSec << ',' << observation.Score << '\n';
    }
    return out.str();
}

static std::string evidence_ingredient_grade_json(const EvidenceIngredientReport& report)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(6);
    out << "{\n"
        << "  \"available\": " << (report.available ? "true" : "false") << ",\n"
        << "  \"preroll_ms\": " << report.preroll_ms << ",\n"
        << "  \"postroll_ms\": " << report.postroll_ms << ",\n"
        << "  \"types\": {\n";
    size_t emitted = 0;
    for (const auto& [type, stats] : report.by_type)
    {
        out << "    \"" << type << "\": {"
            << "\"target_count\": " << stats.target_count << ", "
            << "\"observation_count\": " << stats.observation_count << ", "
            << "\"matched_count\": " << stats.matched_count << ", "
            << "\"precision\": " << stats.precision << ", "
            << "\"recall\": " << stats.recall << ", "
            << "\"f1\": "
            << (stats.precision + stats.recall > 0.0
                ? 2.0 * stats.precision * stats.recall / (stats.precision + stats.recall)
                : 0.0) << ", "
            << "\"mean_abs_error_ms\": " << stats.mean_abs_error_ms << ", "
            << "\"mean_decision_latency_ms\": " << stats.mean_decision_latency_ms << ", "
            << "\"region_count\": " << stats.region_count << ", "
            << "\"exact_region_count\": " << stats.exact_region_count << ", "
            << "\"region_count_abs_error\": " << stats.region_count_abs_error << ", "
            << "\"clustered_120_target_count\": " << stats.clustered_120_target_count << ", "
            << "\"clustered_120_matched_count\": " << stats.clustered_120_matched_count << ", "
            << "\"clustered_150_target_count\": " << stats.clustered_150_target_count << ", "
            << "\"clustered_150_matched_count\": " << stats.clustered_150_matched_count << ", "
            << "\"clustered_200_target_count\": " << stats.clustered_200_target_count << ", "
            << "\"clustered_200_matched_count\": " << stats.clustered_200_matched_count << ", "
            << "\"outside_speech_observation_count\": " << stats.outside_speech_observation_count << "}";
        out << (++emitted < report.by_type.size() ? ",\n" : "\n");
    }
    out << "  }\n}\n";
    return out.str();
}

static SyllableCandidateSetReport grade_syllable_candidate_sets(
    const std::vector<ProsodicPeakTarget>& targets,
    const TArray<FOffgridAIStreamingSyllableCandidateSet>& candidate_sets)
{
    SyllableCandidateSetReport report;
    report.available = !targets.empty();
    std::vector<bool> used_targets(targets.size(), false);
    for (const auto& candidate_set : candidate_sets)
    {
        int best_target = INDEX_NONE;
        double best_error = 0.100001;
        for (size_t target_index = 0; target_index < targets.size(); ++target_index)
        {
            if (used_targets[target_index] || !targets[target_index].has_gold) continue;
            const double error = candidate_set.AudioCenterSec < targets[target_index].gold_start
                ? targets[target_index].gold_start - candidate_set.AudioCenterSec
                : (candidate_set.AudioCenterSec > targets[target_index].gold_end
                    ? candidate_set.AudioCenterSec - targets[target_index].gold_end : 0.0);
            if (error < best_error)
            {
                best_error = error;
                best_target = static_cast<int>(target_index);
            }
        }
        if (best_target == INDEX_NONE)
        {
            ++report.unmatched_pulse_count;
            continue;
        }
        used_targets[static_cast<size_t>(best_target)] = true;
        ++report.matched_pulse_count;
        for (size_t rank = 0; rank < report.hit_counts.size(); ++rank)
        {
            const int32 candidate_limit = FMath::Min(
                static_cast<int32>(rank + 1),
                candidate_set.SyllableIndices.Num());
            bool hit = false;
            for (int32 candidate_index = 0; candidate_index < candidate_limit; ++candidate_index)
            {
                if (candidate_set.SyllableIndices[candidate_index] == best_target)
                {
                    hit = true;
                    break;
                }
            }
            if (hit) ++report.hit_counts[rank];
        }
    }
    return report;
}

static std::string syllable_candidate_sets_csv(
    const TArray<FOffgridAIStreamingSyllableCandidateSet>& candidate_sets)
{
    std::ostringstream out;
    out << "audio_center,decision,rank,syllable_index,score\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& candidate_set : candidate_sets)
    {
        for (int32 rank = 0; rank < candidate_set.SyllableIndices.Num(); ++rank)
        {
            out << candidate_set.AudioCenterSec << ',' << candidate_set.DecisionSec << ','
                << rank + 1 << ',' << candidate_set.SyllableIndices[rank] << ','
                << (candidate_set.Scores.IsValidIndex(rank) ? candidate_set.Scores[rank] : 0.0f)
                << '\n';
        }
    }
    return out.str();
}

static std::string syllable_candidate_set_grade_json(const SyllableCandidateSetReport& report)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(6)
        << "{\n"
        << "  \"available\": " << (report.available ? "true" : "false") << ",\n"
        << "  \"matched_pulse_count\": " << report.matched_pulse_count << ",\n"
        << "  \"unmatched_pulse_count\": " << report.unmatched_pulse_count << ",\n"
        << "  \"recall_at\": {\n";
    for (size_t rank = 0; rank < report.hit_counts.size(); ++rank)
    {
        const double recall = report.matched_pulse_count > 0
            ? static_cast<double>(report.hit_counts[rank]) / report.matched_pulse_count : 0.0;
        out << "    \"" << rank + 1 << "\": {\"hit_count\": "
            << report.hit_counts[rank] << ", \"recall\": " << recall << "}"
            << (rank + 1 < report.hit_counts.size() ? ",\n" : "\n");
    }
    out << "  }\n}\n";
    return out.str();
}

static std::vector<ProsodicPeakObservation> runtime_syllable_assignment_observations(
    const std::vector<ProsodicPeakTarget>& targets,
    const TArray<FOffgridAIRuntimeSyllableAssignmentDiagnosticRow>& rows,
    const std::set<std::string>& included_anchor_kinds = {})
{
    std::map<int, int> target_by_phone_index;
    for (const ProsodicPeakTarget& target : targets)
    {
        target_by_phone_index[target.expected_phone_index] = target.target_index;
    }
    std::vector<ProsodicPeakObservation> observations;
    for (const FOffgridAIRuntimeSyllableAssignmentDiagnosticRow& row : rows)
    {
        if (!included_anchor_kinds.empty()
            && included_anchor_kinds.find(to_std(row.AnchorKind)) == included_anchor_kinds.end())
        {
            continue;
        }
        auto Emit = [&](int32 PhoneIndex, float ObservedAudioSec)
        {
            if (PhoneIndex < 0) return;
            const auto target_it = target_by_phone_index.find(PhoneIndex);
            if (target_it == target_by_phone_index.end()) return;
            ProsodicPeakObservation observation;
            observation.target_index = target_it->second;
            observation.peak_sec = ObservedAudioSec;
            // The causal envelope detector requires 14 future 10 ms frames. The
            // audible playback clock is intentionally behind buffered audio, so it
            // is not the evidence-availability timestamp.
            observation.decision_sec = ObservedAudioSec + 0.145;
            observation.prominence = row.Prominence;
            observation.score = row.Confidence;
            observation.family_score = row.Confidence;
            observations.push_back(observation);
        };
        Emit(row.PhoneIndex, row.ObservedAudioSec);
    }
    return observations;
}

static std::vector<ProsodicPeakTarget> word_start_nucleus_targets(
    const std::vector<ProsodicPeakTarget>& syllable_targets)
{
    std::vector<ProsodicPeakTarget> targets;
    std::set<int> seen_words;
    for (const ProsodicPeakTarget& source : syllable_targets)
    {
        if (source.word_index < 0 || !seen_words.insert(source.word_index).second)
            continue;
        ProsodicPeakTarget target = source;
        target.target_index = static_cast<int>(targets.size());
        targets.push_back(std::move(target));
    }
    return targets;
}

// The active runtime contract is narrower than generic syllable
// assignment: the first actually renderable viseme of each word must be
// centered on that word's own first MFA-aligned nucleus. Grade the performed
// event, not merely the detector row, so playhead clamping, ordering floors, or
// later retiming cannot silently dilute an otherwise correct correspondence.
static std::vector<ProsodicPeakObservation> runtime_word_start_nucleus_observations(
    const std::vector<ProsodicPeakTarget>& targets,
    const TArray<FOffgridAIRuntimeSyllableAssignmentDiagnosticRow>& rows,
    const FOffgridAICommittedVisemeTrack& track)
{
    std::map<int, int> target_by_phone_index;
    for (const ProsodicPeakTarget& target : targets)
        target_by_phone_index[target.expected_phone_index] = target.target_index;

    std::vector<ProsodicPeakObservation> observations;
    for (const FOffgridAIRuntimeSyllableAssignmentDiagnosticRow& row : rows)
    {
        if (row.AnchorKind != FName(TEXT("audio_word_start_prior"))) continue;
        const auto target_it = target_by_phone_index.find(row.PhoneIndex);
        if (target_it == target_by_phone_index.end()) continue;

        const FOffgridAICommittedVisemeEvent* anchor_event = nullptr;
        for (const auto& event : track.Events)
        {
            if (event.WordIndex != row.WordIndex
                || !event.bIsRenderable
                || event.bCanceledByWordHandoff
                || event.AcousticAnchorKind
                    != FName(TEXT("audio_word_start_nucleus")))
                continue;
            if (anchor_event == nullptr
                || FMath::Abs(event.AcousticAnchorErrorSeconds)
                    < FMath::Abs(anchor_event->AcousticAnchorErrorSeconds))
                anchor_event = &event;
        }
        if (anchor_event == nullptr) continue;

        ProsodicPeakObservation observation;
        observation.target_index = target_it->second;
        observation.peak_sec = anchor_event->FinalRenderCenterSeconds;
        observation.decision_sec = row.ObservedAudioSec + 0.145;
        observation.prominence = row.Prominence;
        observation.score = row.Confidence;
        observation.family_score = row.Confidence;
        observations.push_back(observation);
    }
    return observations;
}

static std::vector<ProsodicPeakObservation> performed_word_head_anchor_observations(
    const FOffgridAICommittedVisemeTrack& track)
{
    std::map<int32, const FOffgridAICommittedVisemeEvent*> anchor_by_word;
    for (const auto& event : track.Events)
    {
        if (event.WordIndex < 0
            || !event.bIsRenderable
            || event.bCanceledByWordHandoff
            || event.AcousticAnchorKind
                != FName(TEXT("audio_word_start_nucleus")))
            continue;
        const auto existing = anchor_by_word.find(event.WordIndex);
        if (existing == anchor_by_word.end()
            || FMath::Abs(event.AcousticAnchorErrorSeconds)
                < FMath::Abs(existing->second->AcousticAnchorErrorSeconds))
            anchor_by_word[event.WordIndex] = &event;
    }

    std::vector<ProsodicPeakObservation> observations;
    for (const auto& item : anchor_by_word)
    {
        ProsodicPeakObservation observation;
        observation.peak_sec = item.second->FinalRenderCenterSeconds;
        observation.decision_sec = item.second->CommitPlaybackSeconds;
        observation.score = 1.0;
        observation.family_score = 1.0;
        observations.push_back(observation);
    }
    return observations;
}

static std::string runtime_syllable_anchor_diagnostics_csv(
    const TArray<FOffgridAIRuntimeSyllableAssignmentDiagnosticRow>& rows)
{
    std::ostringstream out;
    out << "line_id,update_ordinal,audio_speech_region_index,text_speech_region_index,nucleus_phone_index,pulse_audio_sec,prominence,confidence,skip_count,anchor_kind,timeline_correction_sec,word_index,word_prior_rate,observed_word_interval_sec,prior_word_interval_sec,canceled_prior_word_event_count,nucleus_audio_sec,visual_anchor_audio_sec,visual_anchor_kind,visual_anchor_phone_index\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& row : rows)
    {
        out << to_std(row.LineID) << ','
            << row.UpdateOrdinal << ','
            << row.AudioSpeechRegionIndex << ','
            << row.TextSpeechRegionIndex << ','
            << row.PhoneIndex << ','
            << row.ObservedAudioSec << ','
            << row.Prominence << ','
            << row.Confidence << ','
            << row.SkipCount << ','
            << to_std(row.AnchorKind) << ','
            << row.TimelineCorrectionSec << ','
            << row.WordIndex << ','
            << row.WordPriorRate << ','
            << row.ObservedWordIntervalSec << ','
            << row.PriorWordIntervalSec << ','
            << row.CanceledPriorWordEventCount << ','
            << row.NucleusAudioSec << ','
            << row.VisualAnchorAudioSec << ','
            << to_std(row.VisualAnchorKind) << ','
            << row.VisualAnchorPhoneIndex << '\n';
    }
    return out.str();
}


static std::string prosodic_peak_grade_json(const ProsodicPeakReport& report)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(6)
        << "{\n  \"available\": " << (report.available ? "true" : "false") << ",\n"
        << "  \"target_count\": " << report.target_count << ",\n"
        << "  \"observation_count\": " << report.observation_count << ",\n"
        << "  \"matched_count\": " << report.matched_count << ",\n"
        << "  \"precision\": " << report.precision << ",\n"
        << "  \"recall\": " << report.recall << ",\n"
        << "  \"mean_abs_error_ms\": " << report.mean_abs_error_ms << ",\n"
        << "  \"median_abs_error_ms\": " << report.median_abs_error_ms << ",\n"
        << "  \"p90_abs_error_ms\": " << report.p90_abs_error_ms << ",\n"
        << "  \"mean_decision_latency_ms\": " << report.mean_decision_latency_ms << ",\n"
        << "  \"raw_observation_count\": " << report.raw_observation_count << ",\n"
        << "  \"raw_ungradeable_observation_count\": " << report.raw_ungradeable_observation_count << ",\n"
        << "  \"raw_matched_count\": " << report.raw_matched_count << ",\n"
        << "  \"raw_precision\": " << report.raw_precision << ",\n"
        << "  \"raw_recall\": " << report.raw_recall << ",\n"
        << "  \"raw_mean_abs_error_ms\": " << report.raw_mean_abs_error_ms << "\n}\n";
    return out.str();
}

static std::string audio_landmark_frames_csv(const TArray<FOffgridAIStreamingAudioFeatureFrame>& frames)
{
    std::ostringstream out;
    out << "frame_index,start,end,center,mbp,fv,w,chjjsh,round,comma_lull,"
        << "rich_low,rich_mid,rich_high,rich_centroid,rich_rolloff,rich_flatness,rich_flux,rich_periodicity,"
        << "top_landmark,top_score\n";
    out << std::fixed << std::setprecision(6);

    for (int32 i = 0; i < frames.Num(); ++i)
    {
        const auto& frame = frames[i];
        const FOffgridAIArticulatoryProbabilityField field = FOffgridAIAcousticEvidence::BuildArticulatoryProbabilityField(frame);
        const double mbp = landmark_score_for_type("mbp", field);
        const double fv = landmark_score_for_type("fv", field);
        const double w = landmark_score_for_type("w", field);
        const double chjjsh = landmark_score_for_type("chjjsh", field);
        const double round = landmark_score_for_type("round", field);
        const double comma_lull = comma_lull_score_for_frame(frame);

        std::string top_type = "none";
        double top_score = mbp;
        top_type = "mbp";
        if (fv > top_score) { top_score = fv; top_type = "fv"; }
        if (w > top_score) { top_score = w; top_type = "w"; }
        if (chjjsh > top_score) { top_score = chjjsh; top_type = "chjjsh"; }
        if (round > top_score) { top_score = round; top_type = "round"; }
        if (comma_lull > top_score) { top_score = comma_lull; top_type = "comma_lull"; }

        out << i << ','
            << frame.AudioBufferStartSec << ','
            << frame.AudioBufferEndSec << ','
            << frame.AudioBufferCenterSec << ','
            << mbp << ','
            << fv << ','
            << w << ','
            << chjjsh << ','
            << round << ','
            << comma_lull << ','
            << frame.RichLowBandNorm << ','
            << frame.RichMidBandNorm << ','
            << frame.RichHighBandNorm << ','
            << frame.RichSpectralCentroidNorm << ','
            << frame.RichSpectralRolloffNorm << ','
            << frame.RichSpectralFlatness << ','
            << frame.RichSpectralFlux << ','
            << frame.RichPeriodicity << ','
            << top_type << ','
            << top_score << '\n';
    }

    return out.str();
}


static std::string gold_visible_visemes_csv(const std::vector<HandmadeLabel>& labels)
{
    std::ostringstream out;
    out << "start,end,pose,word,confidence,word_index,speech_region_index,text_sentence_index,source_phone_global_index,source_phone_word_index,source_phone_base\n";
    out << std::fixed << std::setprecision(6);
    for (const HandmadeLabel& label : labels)
    {
        out << label.start << ','
            << label.end << ','
            << label.pose << ','
            << label.word << ','
            << label.confidence << ','
            << label.word_index << ','
            << label.speech_region_index << ','
            << label.sentence_index << ','
            << label.source_phone_global_index << ','
            << label.source_phone_word_index << ','
            << label.source_phone_base << '\n';
    }
    return out.str();
}

static std::vector<HandmadeLabel> build_gold_visible_labels(
    const FOffgridAITextVisemePlan& plan,
    const std::vector<GoldPhoneTiming>& gold_phones)
{
    std::vector<HandmadeLabel> out;
    if (plan.Events.Num() <= 0 || gold_phones.empty())
    {
        return out;
    }

    std::vector<const GoldPhoneTiming*> gold_by_plan_phone(
        static_cast<size_t>(plan.ExpectedPhones.Num()),
        nullptr);

    // Align each word's phone sequences independently. MFA pronunciations can
    // contain an insertion or deletion (for example an onset /HH/ before
    // "what"), so neither corpus-global nor word-local ordinal indices are a
    // safe identity contract after the first difference.
    for (int32 word_index = 0; word_index < plan.WordPhoneBeginIndices.Num(); ++word_index)
    {
        const int32 plan_begin = plan.WordPhoneBeginIndices[word_index];
        const int32 plan_end = plan.WordPhoneEndIndices.IsValidIndex(word_index)
            ? plan.WordPhoneEndIndices[word_index]
            : plan_begin;
        std::vector<const GoldPhoneTiming*> word_gold;
        for (const GoldPhoneTiming& phone : gold_phones)
        {
            if (phone.word_index == word_index) word_gold.push_back(&phone);
        }
        std::stable_sort(
            word_gold.begin(),
            word_gold.end(),
            [](const GoldPhoneTiming* a, const GoldPhoneTiming* b)
            {
                return a->word_phone_index < b->word_phone_index;
            });

        const int n = std::max(0, plan_end - plan_begin);
        const int m = static_cast<int>(word_gold.size());
        std::vector<int> dp(static_cast<size_t>((n + 1) * (m + 1)), 0);
        std::vector<uint8_t> parent(static_cast<size_t>((n + 1) * (m + 1)), 0);
        auto at = [&](int i, int j) -> int&
        {
            return dp[static_cast<size_t>(i) * static_cast<size_t>(m + 1) + static_cast<size_t>(j)];
        };
        auto parent_at = [&](int i, int j) -> uint8_t&
        {
            return parent[static_cast<size_t>(i) * static_cast<size_t>(m + 1) + static_cast<size_t>(j)];
        };

        for (int i = 1; i <= n; ++i)
        {
            for (int j = 1; j <= m; ++j)
            {
                const std::string expected_base = to_std(plan.ExpectedPhones[plan_begin + i - 1].BasePhone);
                if (expected_base == word_gold[static_cast<size_t>(j - 1)]->phone_base)
                {
                    at(i, j) = at(i - 1, j - 1) + 1;
                    parent_at(i, j) = 3;
                }
                else if (at(i - 1, j) >= at(i, j - 1))
                {
                    at(i, j) = at(i - 1, j);
                    parent_at(i, j) = 1;
                }
                else
                {
                    at(i, j) = at(i, j - 1);
                    parent_at(i, j) = 2;
                }
            }
        }

        int i = n;
        int j = m;
        while (i > 0 && j > 0)
        {
            const uint8_t step = parent_at(i, j);
            if (step == 3)
            {
                gold_by_plan_phone[static_cast<size_t>(plan_begin + i - 1)] = word_gold[static_cast<size_t>(j - 1)];
                --i;
                --j;
            }
            else if (step == 1)
            {
                --i;
            }
            else
            {
                --j;
            }
        }
    }

    for (const FOffgridAITextVisemeEvent& event : plan.Events)
    {
        if (!event.bIsRenderable) continue;
        const GoldPhoneTiming* matched_phone = nullptr;
        if (event.SourcePhoneGlobalIndex >= 0
            && static_cast<size_t>(event.SourcePhoneGlobalIndex) < gold_by_plan_phone.size())
        {
            matched_phone = gold_by_plan_phone[static_cast<size_t>(event.SourcePhoneGlobalIndex)];
        }

        if (!matched_phone)
        {
            continue;
        }

        HandmadeLabel label;
        label.start = matched_phone->start;
        label.end = matched_phone->end;
        label.pose = to_std(event.PoseID);
        label.word = matched_phone->word.empty() ? to_std(event.SourceText) : matched_phone->word;
        label.confidence = 1.0;
        label.word_index = matched_phone->word_index;
        label.speech_region_index = matched_phone->speech_region_index;
        label.sentence_index = matched_phone->sentence_index;
        label.source_phone_global_index = event.SourcePhoneGlobalIndex;
        label.source_phone_word_index = event.SourcePhoneIndex;
        label.source_phone_base = to_std(event.SourcePhoneBase);
        out.push_back(label);
    }

    std::stable_sort(
        out.begin(),
        out.end(),
        [](const HandmadeLabel& a, const HandmadeLabel& b)
        {
            if (a.start != b.start) return a.start < b.start;
            if (a.word_index != b.word_index) return a.word_index < b.word_index;
            return a.pose < b.pose;
        });
    return out;
}

static std::string planned_csv(const FOffgridAITextVisemePlan& plan)
{
    std::ostringstream out;
    out << "index,pose,word,word_index,text_sentence_index,text_center_norm,strength,jaw_open_target,source_phone,source_phone_index,visual_role,renderable,generator\n";
    out << std::fixed << std::setprecision(6);
    for (int32 i = 0; i < plan.Events.Num(); ++i)
    {
        const auto& event = plan.Events[i];
        const double center = static_cast<double>(event.StartNorm + event.EndNorm) * 0.5;
        out << i << ','
            << to_std(event.PoseID) << ','
            << to_std(event.SourceText) << ','
            << event.WordIndex << ','
            << event.SentenceIndex << ','
            << center << ','
            << event.Strength << ','
            << event.JawOpenTarget << ','
            << to_std(event.SourcePhoneBase) << ','
            << event.SourcePhoneIndex << ','
            << visual_phone_role_name(event.VisualRole) << ','
            << (event.bIsRenderable ? 1 : 0) << ','
            << to_std(event.Generator) << '\n';
    }
    return out.str();
}

static std::string planned_words_csv(const FOffgridAITextVisemePlan& plan)
{
    std::ostringstream out;
    out << "word_index,word,text_sentence_index,phone_begin_index,phone_end_index,visible_event_begin_index,visible_event_end_index,boundary_mark,pause_class,pause_seconds\n";
    for (int32 word_index = 0; word_index < plan.WordPhoneBeginIndices.Num(); ++word_index)
    {
        std::string word;
        if (word_index >= 0)
        {
            const int32 phone_begin = plan.WordPhoneBeginIndices[word_index];
            if (phone_begin >= 0 && phone_begin < plan.ExpectedPhones.Num())
            {
                word = to_std(plan.ExpectedPhones[phone_begin].SourceWord);
            }
        }
        const TCHAR boundary_mark = plan.WordBoundaryPunctuationAfter.IsValidIndex(word_index)
            ? plan.WordBoundaryPunctuationAfter[word_index]
            : TCHAR(0);
        out << word_index << ','
            << word << ','
            << (plan.WordSentenceIndices.IsValidIndex(word_index) ? plan.WordSentenceIndices[word_index] : 0) << ','
            << plan.WordPhoneBeginIndices[word_index] << ','
            << plan.WordPhoneEndIndices[word_index] << ','
            << (plan.WordVisibleEventBeginIndices.IsValidIndex(word_index) ? plan.WordVisibleEventBeginIndices[word_index] : INDEX_NONE) << ','
            << (plan.WordVisibleEventEndIndices.IsValidIndex(word_index) ? plan.WordVisibleEventEndIndices[word_index] : INDEX_NONE) << ','
            << static_cast<int>(boundary_mark) << ','
            << pause_class_name(plan.WordBoundaryPauseClassAfter.IsValidIndex(word_index)
                ? plan.WordBoundaryPauseClassAfter[word_index]
                : EOffgridAIBoundaryPauseClass::None) << ','
            << (plan.WordBoundaryPauseSecondsAfter.IsValidIndex(word_index)
                ? plan.WordBoundaryPauseSecondsAfter[word_index]
                : 0.0f)
            << '\n';
    }
    return out.str();
}

static std::string planned_boundaries_csv(const FOffgridAITextVisemePlan& plan)
{
    std::ostringstream out;
    out << "word_index,next_word_index,boundary_mark,pause_class,pause_seconds\n";
    out << std::fixed << std::setprecision(6);
    const int32 boundary_count = std::min(plan.WordBoundaryPunctuationAfter.Num(), plan.WordBoundaryPauseClassAfter.Num());
    for (int32 word_index = 0; word_index < boundary_count; ++word_index)
    {
        const TCHAR boundary_mark = plan.WordBoundaryPunctuationAfter[word_index];
        const EOffgridAIBoundaryPauseClass pause_class = plan.WordBoundaryPauseClassAfter[word_index];
        out << word_index << ','
            << (word_index + 1) << ','
            << static_cast<int>(boundary_mark) << ','
            << pause_class_name(pause_class) << ','
            << (plan.WordBoundaryPauseSecondsAfter.IsValidIndex(word_index)
                ? plan.WordBoundaryPauseSecondsAfter[word_index]
                : 0.0f)
            << '\n';
    }
    return out.str();
}

static std::string expected_phones_csv(const FOffgridAITextVisemePlan& plan)
{
    std::ostringstream out;
    out << "phone_index,word_phone_index,phone,base_phone,word,word_index,speech_region_index,text_sentence_index,is_vowel,is_visible_viseme,visual_role,first_visible_event_index,pronunciation_variant_index,pronunciation_variant_count,uses_fallback_pronunciation,boundary_after_word,pause_class_after_word,pause_seconds_after_word,weight_seconds\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& phone : plan.ExpectedPhones)
    {
        out << phone.PhoneIndex << ','
            << phone.WordPhoneIndex << ','
            << to_std(phone.Phone) << ','
            << to_std(phone.BasePhone) << ','
            << to_std(phone.SourceWord) << ','
            << phone.WordIndex << ','
            << phone.SpeechRegionIndex << ','
            << phone.SentenceIndex << ','
            << (phone.bIsVowel ? 1 : 0) << ','
            << (phone.bIsVisibleViseme ? 1 : 0) << ','
            << visual_phone_role_name(phone.VisualRole) << ','
            << phone.FirstVisibleEventIndex << ','
            << phone.PronunciationVariantIndex << ','
            << phone.PronunciationVariantCount << ','
            << (phone.bUsesFallbackPronunciation ? 1 : 0) << ','
            << static_cast<int>(phone.BoundaryAfterWord) << ','
            << pause_class_name(
                plan.WordBoundaryPauseClassAfter.IsValidIndex(phone.WordIndex)
                    ? plan.WordBoundaryPauseClassAfter[phone.WordIndex]
                    : EOffgridAIBoundaryPauseClass::None) << ','
            << (plan.WordBoundaryPauseSecondsAfter.IsValidIndex(phone.WordIndex)
                ? plan.WordBoundaryPauseSecondsAfter[phone.WordIndex]
                : 0.0f) << ','
            << phone.WeightSeconds << '\n';
    }
    return out.str();
}

static std::string planned_syllables_csv(const FOffgridAITextVisemePlan& plan)
{
    std::ostringstream out;
    out << "syllable_index,word_syllable_index,word,word_index,speech_region_index,text_sentence_index,phone_begin_index,phone_end_index,nucleus_phone_index,nucleus_phone,nucleus_base_phone\n";
    for (const FOffgridAIPlannedSyllable& syllable : plan.Syllables)
    {
        const FOffgridAIExpectedPhone* nucleus = plan.ExpectedPhones.IsValidIndex(syllable.NucleusPhoneIndex)
            ? &plan.ExpectedPhones[syllable.NucleusPhoneIndex]
            : nullptr;
        out << syllable.SyllableIndex << ','
            << syllable.WordSyllableIndex << ','
            << (nucleus ? to_std(nucleus->SourceWord) : std::string()) << ','
            << syllable.WordIndex << ','
            << syllable.SpeechRegionIndex << ','
            << syllable.SentenceIndex << ','
            << syllable.PhoneBeginIndex << ','
            << syllable.PhoneEndIndex << ','
            << syllable.NucleusPhoneIndex << ','
            << (nucleus ? to_std(nucleus->Phone) : std::string()) << ','
            << (nucleus ? to_std(nucleus->BasePhone) : std::string()) << '\n';
    }
    return out.str();
}

static std::string speech_csv(const TArray<FOffgridAIStreamingSpeechRegion>& speech_regions)
{
    std::ostringstream out;
    out << "index,start,end,last_speech,started,ended,provisional_end,end_decision,reopen_count,end_reason\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& speech_region : speech_regions)
    {
        out << speech_region.SpeechRegionIndex << ','
            << speech_region.AudioBufferStartSec << ','
            << speech_region.AudioBufferEndSec << ','
            << speech_region.AudioBufferLastSpeechSec << ','
            << (speech_region.bStarted ? 1 : 0) << ','
            << (speech_region.bEnded ? 1 : 0) << ','
            << speech_region.ProvisionalEndSec << ','
            << speech_region.EndDecisionSec << ','
            << speech_region.ReopenCount << ','
            << to_std(speech_region.EndReason) << '\n';
    }
    return out.str();
}

static std::string occupancy_frames_csv(const TArray<FOffgridAIStreamingAudioFeatureFrame>& frames)
{
    std::ostringstream out;
    out << "index,start,end,center,rms,rms_norm,delta_rms,flux,zcr,low_band,mid_band,high_band,centroid,periodicity,evidence,open_threshold,close_threshold,in_speech_before,in_speech_after,open_candidate,keep_open,strong_onset,strong_quiet,low_evidence,endpoint_active,silence_accum,endpoint_start,active_speech_region_start,active_speech_region_end,frame_started_speech_region,frame_closed_speech_region,frame_bridged_speech_region,decision,local_rms_peak,local_rms_valley,local_flux_peak,derived_pause_family,derived_pause_confidence,learned_speech_probability,learned_speech\n";
    out << std::fixed << std::setprecision(6);
    for (int32 i = 0; i < frames.Num(); ++i)
    {
        const auto& f = frames[i];
        out << i << ','
            << f.AudioBufferStartSec << ','
            << f.AudioBufferEndSec << ','
            << f.AudioBufferCenterSec << ','
            << f.RMS << ','
            << f.RMSNorm << ','
            << f.DeltaRMS << ','
            << f.Flux << ','
            << f.ZCR << ','
            << f.LowBandNorm << ','
            << f.MidBandNorm << ','
            << f.HighBandNorm << ','
            << f.SpectralCentroidNorm << ','
            << f.Periodicity << ','
            << f.SpeechEvidence << ','
            << f.OpenThreshold << ','
            << f.CloseThreshold << ','
            << (f.bInSpeechBeforeFrame ? 1 : 0) << ','
            << (f.bInSpeechAfterFrame ? 1 : 0) << ','
            << (f.bOpenCandidate ? 1 : 0) << ','
            << (f.bKeepOpen ? 1 : 0) << ','
            << (f.bStrongOnsetAnchor ? 1 : 0) << ','
            << (f.bStrongQuiet ? 1 : 0) << ','
            << (f.bLowEvidence ? 1 : 0) << ','
            << (f.bEndpointCandidateActive ? 1 : 0) << ','
            << f.SilenceAccumSec << ','
            << f.EndpointCandidateStartSec << ','
            << f.ActiveSpeechRegionStartSec << ','
            << f.ActiveSpeechRegionEndSec << ','
            << (f.bFrameStartedSpeechRegion ? 1 : 0) << ','
            << (f.bFrameClosedSpeechRegion ? 1 : 0) << ','
            << (f.bFrameBridgedSpeechRegion ? 1 : 0) << ','
            << to_std(f.OccupancyDecision) << ','
            << (f.bLocalRMSPeak ? 1 : 0) << ','
            << (f.bLocalRMSValley ? 1 : 0) << ','
            << (f.bLocalFluxPeak ? 1 : 0) << ','
            << derived_pause_family(f) << ','
            << derived_pause_confidence(f) << ','
            << f.LearnedSpeechProbability << ','
            << (f.bLearnedSpeech ? 1 : 0) << '\n';
    }
    return out.str();
}

static std::string phone_class_frames_csv(const TArray<FOffgridAIStreamingAudioFeatureFrame>& frames)
{
    std::ostringstream out;
    out << "index,start,end,center,top_class,top_score,silence,vowel_open,vowel_front,vowel_round,bilabial,labiodental,dental,sibilant,stop_burst,liquid,glide,nasal,unknown,speech,voiced,vowel,fricative,closure,release,sonorant,transition,red_herring,low_tilt,high_tilt,spectral_change,energy_change\n";
    out << std::fixed << std::setprecision(6);

    auto top_class_name = [](const FOffgridAIPhoneClassScores& Scores, float& OutScore) -> std::string
    {
        struct Candidate
        {
            EOffgridAIPhoneClass Class = EOffgridAIPhoneClass::Unknown;
            float Score = 0.0f;
        };

        Candidate Best;
        for (int32 ClassIndex = 0; ClassIndex <= static_cast<int32>(EOffgridAIPhoneClass::Unknown); ++ClassIndex)
        {
            const EOffgridAIPhoneClass PhoneClass = static_cast<EOffgridAIPhoneClass>(ClassIndex);
            const float Score = FOffgridAIAcousticEvidence::ScoreForClass(Scores, PhoneClass);
            if (ClassIndex == 0 || Score > Best.Score)
            {
                Best.Class = PhoneClass;
                Best.Score = Score;
            }
        }

        OutScore = Best.Score;
        return to_std(FOffgridAIAcousticEvidence::PhoneClassToString(Best.Class));
    };

    for (int32 i = 0; i < frames.Num(); ++i)
    {
        const auto& f = frames[i];
        const FOffgridAIArticulatoryProbabilityField field = FOffgridAIAcousticEvidence::BuildArticulatoryProbabilityField(f);
        float top_score = 0.0f;
        const std::string top_class = top_class_name(field.PhoneScores, top_score);

        out << i << ','
            << f.AudioBufferStartSec << ','
            << f.AudioBufferEndSec << ','
            << f.AudioBufferCenterSec << ','
            << top_class << ','
            << top_score << ','
            << field.PhoneScores.Silence << ','
            << field.PhoneScores.VowelOpen << ','
            << field.PhoneScores.VowelFront << ','
            << field.PhoneScores.VowelRound << ','
            << field.PhoneScores.Bilabial << ','
            << field.PhoneScores.Labiodental << ','
            << field.PhoneScores.Dental << ','
            << field.PhoneScores.Sibilant << ','
            << field.PhoneScores.StopBurst << ','
            << field.PhoneScores.Liquid << ','
            << field.PhoneScores.Glide << ','
            << field.PhoneScores.Nasal << ','
            << field.PhoneScores.Unknown << ','
            << field.Speech << ','
            << field.Voiced << ','
            << field.Vowel << ','
            << field.Fricative << ','
            << field.Closure << ','
            << field.Release << ','
            << field.Sonorant << ','
            << field.Transition << ','
            << field.RedHerring << ','
            << field.LowTilt << ','
            << field.HighTilt << ','
            << field.SpectralChange << ','
            << field.EnergyChange << '\n';
    }
    return out.str();
}

static std::string gap_candidates_csv(const TArray<FOffgridAIStreamingSpeechGapCandidate>& gaps)
{
    std::ostringstream out;
    out << "gap_index,prev_speech_region_index,next_speech_region_index,gap_start,gap_end,gap_duration,prev_speech_region_duration,quiet_evidence,quiet_rms_norm,gap_frames,low_evidence_frames,strong_quiet_frames,mean_evidence,mean_rms_norm,mean_periodicity,mean_flux,mean_centroid,min_evidence,min_rms_norm,max_periodicity,max_flux,reopen_evidence,reopen_flux,strong_quiet_close,strong_onset_reopen,bridged,close_reason,decision_class\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& gap : gaps)
    {
        const float inv_frames = gap.GapFrameCount > 0 ? 1.0f / static_cast<float>(gap.GapFrameCount) : 0.0f;
        out << gap.GapIndex << ','
            << gap.PrevSpeechRegionIndex << ','
            << gap.NextSpeechRegionIndex << ','
            << gap.GapStartSec << ','
            << gap.GapEndSec << ','
            << gap.GapDurationSec << ','
            << gap.PrevSpeechRegionDurationSec << ','
            << gap.QuietEvidence << ','
            << gap.QuietRMSNorm << ','
            << gap.GapFrameCount << ','
            << gap.LowEvidenceFrameCount << ','
            << gap.StrongQuietFrameCount << ','
            << gap.GapEvidenceSum * inv_frames << ','
            << gap.GapRMSNormSum * inv_frames << ','
            << gap.GapPeriodicitySum * inv_frames << ','
            << gap.GapFluxSum * inv_frames << ','
            << gap.GapCentroidSum * inv_frames << ','
            << gap.GapEvidenceMin << ','
            << gap.GapRMSNormMin << ','
            << gap.GapPeriodicityMax << ','
            << gap.GapFluxMax << ','
            << gap.ReopenEvidence << ','
            << gap.ReopenFlux << ','
            << (gap.bStrongQuietClose ? 1 : 0) << ','
            << (gap.bStrongOnsetReopen ? 1 : 0) << ','
            << (gap.bBridged ? 1 : 0) << ','
            << to_std(gap.CloseReason) << ','
            << to_std(gap.DecisionClass) << '\n';
    }
    return out.str();
}

static std::string soft_lull_candidates_csv(const TArray<FOffgridAIStreamingSoftLullCandidate>& lulls)
{
    std::ostringstream out;
    out << "lull_index,speech_region_index,lull_start,lull_end,lull_duration,frame_count,mean_relative_rms,min_relative_rms,mean_evidence,min_evidence,reopen_evidence,reopen_flux,strong_onset_reopen\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& lull : lulls)
    {
        const float inv_frames = lull.FrameCount > 0 ? 1.0f / static_cast<float>(lull.FrameCount) : 0.0f;
        out << lull.LullIndex << ','
            << lull.SpeechRegionIndex << ','
            << lull.LullStartSec << ','
            << lull.LullEndSec << ','
            << lull.LullDurationSec << ','
            << lull.FrameCount << ','
            << lull.RelativeRMSSum * inv_frames << ','
            << lull.RelativeRMSMin << ','
            << lull.EvidenceSum * inv_frames << ','
            << lull.EvidenceMin << ','
            << lull.ReopenEvidence << ','
            << lull.ReopenFlux << ','
            << (lull.bStrongOnsetReopen ? 1 : 0) << '\n';
    }
    return out.str();
}

static std::string committed_csv(const FOffgridAIAlignedVisemeTrack& track)
{
    std::ostringstream out;
    out << "index,start,center,end,pose,word,word_index,speech_region_index,text_sentence_index,strength,jaw_open_target,renderable,canceled_by_word_handoff,reason,source_phone_index,source_phone_base,source_phone_class,text_center_norm,text_diagnostic_center,prior_start,prior_center,prior_end,lead_adjusted_center,playback_offset,total_paused_at_commit,min_live_lead_delay,inter_event_floor_delay,total_center_delay,commit_playback,commit_lead,mapped_to_observed_speech,used_initial_speech_anchor,used_resume_anchor,acoustic_anchor_kind,acoustic_anchor_seconds,acoustic_anchor_error_seconds,observed_pause_decay_seconds,observed_resume_onset_seconds,observed_resume_energy_anchor_seconds,boundary_word_index,boundary_mark,boundary_outcome\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& event : track.Events)
    {
        out << event.EventIndex << ','
            << event.RenderStartSeconds << ','
            << event.FinalRenderCenterSeconds << ','
            << event.RenderEndSeconds << ','
            << to_std(event.PoseID) << ','
            << to_std(event.SourceWord) << ','
            << event.WordIndex << ','
            << event.SpeechRegionIndex << ','
            << event.SentenceIndex << ','
            << event.Strength << ','
            << event.JawOpenTarget << ','
            << (event.bIsRenderable ? 1 : 0) << ','
            << (event.bCanceledByWordHandoff ? 1 : 0) << ','
            << to_std(event.CommitReason) << ','
            << event.SourcePhoneIndex << ','
            << to_std(event.SourcePhoneBase) << ','
            << to_std(event.SourcePhoneClass) << ','
            << event.TextCenterNorm << ','
            << event.TextDiagnosticCenterSeconds << ','
            << event.PriorStartSeconds << ','
            << event.PriorCenterSeconds << ','
            << event.PriorEndSeconds << ','
            << event.LeadAdjustedCenterSeconds << ','
            << event.PlaybackOffsetSeconds << ','
            << event.TotalPausedSecondsAtCommit << ','
            << event.MinLiveLeadDelaySeconds << ','
            << event.InterEventFloorDelaySeconds << ','
            << event.TotalCenterDelaySeconds << ','
            << event.CommitPlaybackSeconds << ','
            << event.CommitLeadSeconds << ','
            << (event.bMappedToObservedSpeech ? 1 : 0) << ','
            << (event.bUsedInitialSpeechAnchor ? 1 : 0) << ','
            << (event.bUsedResumeAnchor ? 1 : 0) << ','
            << to_std(event.AcousticAnchorKind) << ','
            << event.AcousticAnchorSeconds << ','
            << event.AcousticAnchorErrorSeconds << ','
            << event.ObservedPauseDecaySeconds << ','
            << event.ObservedResumeOnsetSeconds << ','
            << event.ObservedResumeEnergyAnchorSeconds << ','
            << event.BoundaryWordIndex << ','
            << csv_value(to_std(event.BoundaryMark)) << ','
            << to_std(event.BoundaryOutcome) << '\n';
    }
    return out.str();
}

static std::string dropped_csv(const FOffgridAIAlignedVisemeTrack& track)
{
    (void)track;
    std::ostringstream out;
    out << "index,pose,word,word_index,speech_region_index,text_sentence_index,strong_visible,source_phone_index,source_phone_base,source_phone_class,drop_playback,region_start,region_end,drop_reason\n";
    return out.str();
}

static std::string region_drop_diagnostics_csv(
    const FOffgridAIAlignedVisemeTrack& track,
    const TArray<FOffgridAIStreamingSpeechRegion>& speech_regions)
{
    std::ostringstream out;
    out << "speech_region_index,region_start,region_end,region_duration_ms,committed_event_count,first_committed_center,last_committed_center,last_committed_end,last_commit_to_region_end_ms,underrun_idle_tail_ms\n";
    out << std::fixed << std::setprecision(6);

    int32 max_region_index = speech_regions.Num() - 1;
    for (const auto& event : track.Events)
    {
        if (!event.bIsRenderable || event.bCanceledByWordHandoff) continue;
        max_region_index = std::max(max_region_index, event.SpeechRegionIndex);
    }

    for (int32 region_index = 0; region_index <= max_region_index; ++region_index)
    {
        double region_start = -1.0;
        double region_end = -1.0;
        if (speech_regions.IsValidIndex(region_index))
        {
            region_start = speech_regions[region_index].AudioBufferStartSec;
            region_end = speech_regions[region_index].AudioBufferEndSec;
        }

        int committed_event_count = 0;
        double first_committed_center = -1.0;
        double last_committed_center = -1.0;
        double last_committed_end = -1.0;

        for (const auto& event : track.Events)
        {
            if (!event.bIsRenderable || event.bCanceledByWordHandoff) continue;
            if (event.SpeechRegionIndex != region_index) continue;
            ++committed_event_count;
            const double center = static_cast<double>(event.FinalRenderCenterSeconds);
            const double render_end = static_cast<double>(event.RenderEndSeconds);
            if (first_committed_center < 0.0 || center < first_committed_center) first_committed_center = center;
            if (center > last_committed_center) last_committed_center = center;
            if (render_end > last_committed_end) last_committed_end = render_end;
        }

        if (committed_event_count <= 0) continue;

        const double region_duration_ms = (region_start >= 0.0 && region_end >= region_start)
            ? (region_end - region_start) * 1000.0
            : 0.0;
        const double last_commit_to_region_end_ms = (last_committed_center >= 0.0 && region_end >= 0.0)
            ? (region_end - last_committed_center) * 1000.0
            : 0.0;
        const double underrun_idle_tail_ms = (
            last_committed_end >= 0.0
            && region_end >= last_committed_end)
            ? (region_end - last_committed_end) * 1000.0
            : 0.0;

        out << region_index << ','
            << region_start << ','
            << region_end << ','
            << region_duration_ms << ','
            << committed_event_count << ','
            << first_committed_center << ','
            << last_committed_center << ','
            << last_committed_end << ','
            << last_commit_to_region_end_ms << ','
            << underrun_idle_tail_ms
            << '\n';
    }

    return out.str();
}

static std::string commit_decisions_csv(const FOffgridAIAlignedVisemeTrack& track)
{
    std::ostringstream out;
    out << "event_index,pose,word,word_index,source_phone_index,source_phone_base,source_phone_class,renderable,canceled_by_word_handoff,commit_reason,render_start,render_center,render_end,prior_start,prior_center,prior_end,lead_adjusted_center,playback_offset,total_paused_at_commit,min_live_lead_delay,inter_event_floor_delay,total_center_delay,commit_playback,commit_lead,mapped_to_observed_speech,used_initial_speech_anchor,used_resume_anchor,acoustic_anchor_kind,acoustic_anchor_seconds,acoustic_anchor_error_seconds,observed_pause_decay_seconds,observed_resume_onset_seconds,observed_resume_energy_anchor_seconds,boundary_word_index,boundary_mark,boundary_outcome\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& event : track.Events)
    {
        out << event.EventIndex << ','
            << to_std(event.PoseID) << ','
            << to_std(event.SourceWord) << ','
            << event.WordIndex << ','
            << event.SourcePhoneIndex << ','
            << to_std(event.SourcePhoneBase) << ','
            << to_std(event.SourcePhoneClass) << ','
            << (event.bIsRenderable ? 1 : 0) << ','
            << (event.bCanceledByWordHandoff ? 1 : 0) << ','
            << to_std(event.CommitReason) << ','
            << event.RenderStartSeconds << ','
            << event.FinalRenderCenterSeconds << ','
            << event.RenderEndSeconds << ','
            << event.PriorStartSeconds << ','
            << event.PriorCenterSeconds << ','
            << event.PriorEndSeconds << ','
            << event.LeadAdjustedCenterSeconds << ','
            << event.PlaybackOffsetSeconds << ','
            << event.TotalPausedSecondsAtCommit << ','
            << event.MinLiveLeadDelaySeconds << ','
            << event.InterEventFloorDelaySeconds << ','
            << event.TotalCenterDelaySeconds << ','
            << event.CommitPlaybackSeconds << ','
            << event.CommitLeadSeconds << ','
            << (event.bMappedToObservedSpeech ? 1 : 0) << ','
            << (event.bUsedInitialSpeechAnchor ? 1 : 0) << ','
            << (event.bUsedResumeAnchor ? 1 : 0) << ','
            << to_std(event.AcousticAnchorKind) << ','
            << event.AcousticAnchorSeconds << ','
            << event.AcousticAnchorErrorSeconds << ','
            << event.ObservedPauseDecaySeconds << ','
            << event.ObservedResumeOnsetSeconds << ','
            << event.ObservedResumeEnergyAnchorSeconds << ','
            << event.BoundaryWordIndex << ','
            << csv_value(to_std(event.BoundaryMark)) << ','
            << to_std(event.BoundaryOutcome) << '\n';
    }
    return out.str();
}



static std::string stream_tail_csv(const FOffgridAIStreamTailDiagnosticRow& row)
{
    std::ostringstream out;
    out << "line_id,pcm_chunk_count,pcm_bytes_received,pcm_samples_received,last_sample_rate,last_num_channels,last_chunk_start_sample,last_chunk_end_sample,observed_audio_buffer_end,first_speech_audio_buffer_start,speech_region_count,input_stream_closed,diagnostic_kind\n";
    out << std::fixed << std::setprecision(6);
    out << to_std(row.LineID) << ','
        << row.PCMChunkCount << ','
        << row.PCMBytesReceived << ','
        << row.PCMSamplesReceived << ','
        << row.LastSampleRate << ','
        << row.LastNumChannels << ','
        << row.LastChunkStartSample << ','
        << row.LastChunkEndSample << ','
        << row.ObservedAudioBufferEndSec << ','
        << row.FirstSpeechAudioBufferStartSec << ','
        << row.SpeechRegionCount << ','
        << (row.bInputStreamClosed ? 1 : 0) << ','
        << to_std(row.DiagnosticKind) << '\n';
    return out.str();
}

static std::string runtime_speech_regions_csv(const TArray<FOffgridAIRuntimeSpeechRegionDiagnosticRow>& rows)
{
    std::ostringstream out;
    out << "line_id,update_ordinal,final_replay,current_playback_sec,region_index,region_open_sec,region_close_sec,last_speech_sec,provisional_end_sec,end_decision_sec,reopen_count,started,ended,contains_playback_sec,committed_event_count,close_reason,diagnostic_kind\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& row : rows)
    {
        out << to_std(row.LineID) << ','
            << row.UpdateOrdinal << ','
            << (row.bFinalReplay ? 1 : 0) << ','
            << row.CurrentPlaybackSec << ','
            << row.RegionIndex << ','
            << row.RegionOpenSec << ','
            << row.RegionCloseSec << ','
            << row.LastSpeechSec << ','
            << row.ProvisionalEndSec << ','
            << row.EndDecisionSec << ','
            << row.ReopenCount << ','
            << (row.bStarted ? 1 : 0) << ','
            << (row.bEnded ? 1 : 0) << ','
            << (row.bContainsPlaybackSec ? 1 : 0) << ','
            << row.CommittedEventCount << ','
            << to_std(row.CloseReason) << ','
            << to_std(row.DiagnosticKind) << '\n';
    }
    return out.str();
}

static std::string runtime_boundary_state_csv(const TArray<FOffgridAIRuntimeBoundaryDiagnosticRow>& rows)
{
    std::ostringstream out;
    out << "line_id,update_ordinal,final_replay,current_playback_sec,b_playhead_started,b_audio_speech_active,active_speech_region_index,active_text_speech_region_index,active_region_start_sec,active_region_end_sec,timeline_rate,last_matched_syllable_index,last_matched_syllable_phone_index,last_matched_syllable_audio_sec,last_matched_syllable_confidence,scheduler_next_event_index,scheduler_next_phone_index,scheduler_candidate_center_sec,scheduler_commit_frontier_sec,scheduler_commit_lead_sec,scheduler_block_reason,committed_event_count,diagnostic_kind\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& row : rows)
    {
        out << to_std(row.LineID) << ','
            << row.UpdateOrdinal << ','
            << (row.bFinalReplay ? 1 : 0) << ','
            << row.CurrentPlaybackSec << ','
            << (row.bPlayheadStarted ? 1 : 0) << ','
            << (row.bAudioSpeechActive ? 1 : 0) << ','
            << row.ActiveSpeechRegionIndex << ','
            << row.ActiveTextSpeechRegionIndex << ','
            << row.ActiveRegionStartSec << ','
            << row.ActiveRegionEndSec << ','
            << row.TimelineRate << ','
            << row.LastMatchedSyllableIndex << ','
            << row.LastMatchedSyllablePhoneIndex << ','
            << row.LastMatchedSyllableAudioSec << ','
            << row.LastMatchedSyllableConfidence << ','
            << row.SchedulerNextEventIndex << ','
            << row.SchedulerNextPhoneIndex << ','
            << row.SchedulerCandidateCenterSec << ','
            << row.SchedulerCommitFrontierSec << ','
            << row.SchedulerCommitLeadSec << ','
            << to_std(row.SchedulerBlockReason) << ','
            << row.CommittedEventCount << ','
            << to_std(row.DiagnosticKind) << '\n';
    }
    return out.str();
}

struct MatchState
{
    int matched_count = -1;
    int exact_word_matches = -1;
    double total_center_error_ms = 0.0;
    bool reachable = false;
};

static bool better_match_state(const MatchState& A, const MatchState& B)
{
    if (!A.reachable) return false;
    if (!B.reachable) return true;
    if (A.matched_count != B.matched_count) return A.matched_count > B.matched_count;
    if (A.exact_word_matches != B.exact_word_matches) return A.exact_word_matches > B.exact_word_matches;
    return A.total_center_error_ms < B.total_center_error_ms - 1e-9;
}

static std::vector<GradeMatch> compute_monotonic_matches_subset(
    const FOffgridAIAlignedVisemeTrack& track,
    const std::vector<HandmadeLabel>& handmade,
    const std::vector<int>& label_indices,
    const std::vector<int>& event_indices,
    double max_center_error_ms = 180.0)
{
    const int n = static_cast<int>(label_indices.size());
    const int m = static_cast<int>(event_indices.size());
    std::vector<MatchState> dp(static_cast<size_t>((n + 1) * (m + 1)));
    std::vector<uint8_t> parent(static_cast<size_t>((n + 1) * (m + 1)), 0);
    auto at = [&](int i, int j) -> MatchState& { return dp[static_cast<size_t>(i) * static_cast<size_t>(m + 1) + static_cast<size_t>(j)]; };
    auto parent_at = [&](int i, int j) -> uint8_t& { return parent[static_cast<size_t>(i) * static_cast<size_t>(m + 1) + static_cast<size_t>(j)]; };

    at(0, 0) = MatchState{0, 0, 0.0, true};
    for (int i = 0; i <= n; ++i)
    {
        for (int j = 0; j <= m; ++j)
        {
            const MatchState current = at(i, j);
            if (!current.reachable)
            {
                continue;
            }

            if (i < n)
            {
                MatchState candidate = current;
                if (better_match_state(candidate, at(i + 1, j)))
                {
                    at(i + 1, j) = candidate;
                    parent_at(i + 1, j) = 1;
                }
            }

            if (j < m)
            {
                MatchState candidate = current;
                if (better_match_state(candidate, at(i, j + 1)))
                {
                    at(i, j + 1) = candidate;
                    parent_at(i, j + 1) = 2;
                }
            }

            if (i < n && j < m)
            {
                const HandmadeLabel& label = handmade[static_cast<size_t>(label_indices[static_cast<size_t>(i)])];
                const auto& event = track.Events[event_indices[static_cast<size_t>(j)]];
                const bool source_phone_matches = label.source_phone_global_index < 0
                    || event.SourcePhoneIndex == label.source_phone_global_index;
                if (source_phone_matches && to_std(event.PoseID) == label.pose)
                {
                    const double center = (label.start + label.end) * 0.5;
                    const double err_ms = std::abs(static_cast<double>(event.FinalRenderCenterSeconds) - center) * 1000.0;
                    if (err_ms <= max_center_error_ms)
                    {
                        MatchState candidate = current;
                        candidate.matched_count += 1;
                        candidate.total_center_error_ms += err_ms;
                        if (!label.word.empty() && to_std(event.SourceWord) == label.word)
                        {
                            candidate.exact_word_matches += 1;
                        }
                        if (better_match_state(candidate, at(i + 1, j + 1)))
                        {
                            at(i + 1, j + 1) = candidate;
                            parent_at(i + 1, j + 1) = 3;
                        }
                    }
                }
            }
        }
    }

    int best_j = 0;
    for (int j = 1; j <= m; ++j)
    {
        if (better_match_state(at(n, j), at(n, best_j)))
        {
            best_j = j;
        }
    }

    std::vector<GradeMatch> matches;
    int i = n;
    int j = best_j;
    while (i > 0 || j > 0)
    {
        const uint8_t step = parent_at(i, j);
        if (step == 3)
        {
            matches.push_back(GradeMatch{
                label_indices[static_cast<size_t>(i - 1)],
                event_indices[static_cast<size_t>(j - 1)]});
            --i;
            --j;
        }
        else if (step == 2)
        {
            --j;
        }
        else if (step == 1)
        {
            --i;
        }
        else
        {
            break;
        }
    }

    std::reverse(matches.begin(), matches.end());
    return matches;
}

static std::vector<GradeMatch> compute_monotonic_matches(const FOffgridAIAlignedVisemeTrack& track, const std::vector<HandmadeLabel>& handmade)
{
    std::vector<int> label_indices(handmade.size());
    std::iota(label_indices.begin(), label_indices.end(), 0);
    std::vector<int> event_indices;
    for (int32 i = 0; i < track.Events.Num(); ++i)
    {
        if (track.Events[i].bIsRenderable
            && !track.Events[i].bCanceledByWordHandoff) event_indices.push_back(i);
    }
    return compute_monotonic_matches_subset(track, handmade, label_indices, event_indices);
}

static PredictedNeuralTrackResult build_predicted_neural_track(
    const FOffgridAITextVisemePlan& plan,
    const std::vector<PredictedNeuralTrackEvent>& predictions)
{
    PredictedNeuralTrackResult result;
    const bool has_neural_regions = std::any_of(
        predictions.begin(), predictions.end(), [](const auto& prediction) {
            return prediction.speech_region_index >= 0
                && prediction.speech_region_start_sec >= 0.0
                && prediction.speech_region_end_sec >= prediction.speech_region_start_sec;
        });
    if (has_neural_regions)
    {
        for (const auto& prediction : predictions)
        {
            if (prediction.speech_region_index < 0) continue;
            while (result.speech_regions.Num() <= prediction.speech_region_index)
            {
                FOffgridAIStreamingSpeechRegion region;
                region.SpeechRegionIndex = result.speech_regions.Num();
                result.speech_regions.Add(region);
            }
            auto& region = result.speech_regions[prediction.speech_region_index];
            region.AudioBufferStartSec = static_cast<float>(prediction.speech_region_start_sec);
            region.AudioBufferLastSpeechSec = static_cast<float>(prediction.speech_region_end_sec);
            region.AudioBufferEndSec = region.AudioBufferLastSpeechSec;
            region.ProvisionalEndSec = region.AudioBufferEndSec;
            region.EndDecisionSec = region.AudioBufferEndSec;
            region.bStarted = true;
            region.bEnded = true;
            region.EndReason = FName(TEXT("neural_region_head"));
        }
    }
    float last_center = 0.0f;
    int remaining_visible = static_cast<int>(std::count_if(
        predictions.begin(), predictions.end(), [](const auto& prediction) {
            return !prediction.is_silence && prediction.event_index >= 0;
        }));
    bool region_open = false;
    // Frozen-corpus calibration: shorter between-word silence remains inside
    // one region, while a 120 ms neural silence state owns pause/resume
    // segmentation.
    constexpr float kNeuralPauseSplitSeconds = 0.120f;
    for (const PredictedNeuralTrackEvent& prediction : predictions)
    {
        ++result.prediction_count;
        if (prediction.token_index < 0) continue;
        const float requested_center = static_cast<float>(std::max(0.0, prediction.center_sec));
        const float center = std::max(requested_center, last_center);
        const float duration = static_cast<float>(std::clamp(prediction.duration_sec, 0.010, 0.500));
        const float predicted_start = std::max(0.0f, center - 0.5f * duration);
        const float predicted_end = center + 0.5f * duration;
        if (prediction.is_silence)
        {
            const bool leading_or_trailing = result.accepted_count == 0 || remaining_visible == 0;
            if (!has_neural_regions
                && region_open && (duration >= kNeuralPauseSplitSeconds || leading_or_trailing))
            {
                auto& region = result.speech_regions.Last();
                region.AudioBufferLastSpeechSec = std::max(region.AudioBufferStartSec, predicted_start);
                region.AudioBufferEndSec = region.AudioBufferLastSpeechSec;
                region.ProvisionalEndSec = predicted_start;
                region.EndDecisionSec = predicted_end;
                region.bEnded = true;
                region.EndReason = FName(TEXT("neural_silence_state"));
                region_open = false;
            }
            last_center = center;
            continue;
        }
        if (!plan.Events.IsValidIndex(prediction.event_index)) continue;
        const FOffgridAITextVisemeEvent& source = plan.Events[prediction.event_index];
        if (!source.bIsRenderable) continue;
        --remaining_visible;

        if (has_neural_regions)
        {
            if (prediction.speech_region_index < 0
                || !result.speech_regions.IsValidIndex(prediction.speech_region_index))
                continue;
        }
        else if (!region_open)
        {
            FOffgridAIStreamingSpeechRegion region;
            region.SpeechRegionIndex = result.speech_regions.Num();
            region.AudioBufferStartSec = predicted_start;
            region.AudioBufferLastSpeechSec = predicted_end;
            region.AudioBufferEndSec = predicted_end;
            region.bStarted = true;
            region.bEnded = false;
            result.speech_regions.Add(region);
            region_open = true;
        }
        else
        {
            auto& region = result.speech_regions.Last();
            region.AudioBufferLastSpeechSec = std::max(region.AudioBufferLastSpeechSec, predicted_end);
            region.AudioBufferEndSec = region.AudioBufferLastSpeechSec;
        }

        FOffgridAICommittedVisemeEvent event;
        event.EventIndex = prediction.event_index;
        event.PoseID = source.PoseID;
        event.Strength = source.Strength;
        event.JawOpenTarget = source.JawOpenTarget;
        event.SourceWord = source.SourceText;
        event.WordIndex = source.WordIndex;
        event.SentenceIndex = source.SentenceIndex;
        event.SpeechRegionIndex = has_neural_regions
            ? prediction.speech_region_index
            : result.speech_regions.Last().SpeechRegionIndex;
        event.bIsStrongVisibleEvent = source.bIsStrongVisibleEvent;
        event.bIsRenderable = true;
        event.bCanceledByWordHandoff = false;
        event.TextCenterNorm = source.StartNorm + 0.5f * (source.EndNorm - source.StartNorm);
        event.TextDiagnosticCenterSeconds = center;
        event.PriorCenterSeconds = center;
        event.PriorStartSeconds = std::max(0.0f, center - 0.5f * duration);
        event.PriorEndSeconds = center + 0.5f * duration;
        event.LeadAdjustedCenterSeconds = center;
        event.FinalRenderCenterSeconds = center;
        event.RenderStartSeconds = std::max(0.0f, center - 0.5f * duration);
        event.RenderEndSeconds = center + 0.5f * duration;
        event.SourcePhoneIndex = source.SourcePhoneGlobalIndex;
        event.SourcePhoneBase = source.SourcePhoneBase;
        event.bMappedToObservedSpeech = true;
        event.CommitPlaybackSeconds = std::max(0.0f, center - 0.350f);
        event.CommitLeadSeconds = center - event.CommitPlaybackSeconds;
        event.CommitReason = FName(TEXT("neural_monotonic_alignment"));
        event.AcousticAnchorKind = FName(TEXT("neural_transition"));
        event.AcousticAnchorSeconds = center;
        event.AcousticAnchorErrorSeconds = 0.0f;
        result.track.Events.Add(std::move(event));
        last_center = center;
        ++result.accepted_count;
        if (prediction.fallback_used) ++result.deterministic_fallback_count;
    }
    if (!has_neural_regions && region_open)
    {
        auto& region = result.speech_regions.Last();
        region.bEnded = true;
        region.ProvisionalEndSec = region.AudioBufferEndSec;
        region.EndDecisionSec = region.AudioBufferEndSec;
        region.EndReason = FName(TEXT("neural_sequence_end"));
    }
    for (const auto& region : result.speech_regions)
    {
        FOffgridAICommittedVisemeTrack::FSpeechRegion track_region;
        track_region.SpeechRegionIndex = region.SpeechRegionIndex;
        track_region.StartSeconds = region.AudioBufferStartSec;
        track_region.EndSeconds = region.AudioBufferEndSec;
        track_region.bEnded = region.bEnded;
        result.track.SpeechRegions.Add(track_region);
    }
    if (result.speech_regions.Num() > 0)
    {
        result.track.SpeechStartSeconds = result.speech_regions[0].AudioBufferStartSec;
        result.track.SpeechEndSeconds = result.speech_regions.Last().AudioBufferEndSec;
    }
    return result;
}

static void append_json_number_array(
    std::ostringstream& out, const std::vector<double>& values)
{
    out << '[';
    for (size_t index = 0; index < values.size(); ++index)
    {
        if (index > 0) out << ',';
        out << values[index];
    }
    out << ']';
}

static std::string neural_error_samples_json(
    const GradeReport& grade_report,
    const ProsodicPeakReport& syllable_report)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(3)
        << "{\n  \"schema_version\": 1,\n"
        << "  \"miss_penalties_ms\": {\"viseme\": 180.0, \"region\": 1000.0, "
        << "\"word_onset\": 180.0, \"intra_word\": 180.0, \"syllable\": 100.0},\n";
    auto write_metric = [&](const char* name, const std::vector<double>& samples,
                            int reference_count, int matched_count, bool final) {
        out << "  \"" << name << "\": {\"reference_count\": " << reference_count
            << ", \"matched_count\": " << matched_count << ", \"matched_errors_ms\": ";
        append_json_number_array(out, samples);
        out << '}' << (final ? "\n" : ",\n");
    };
    write_metric("viseme_center", grade_report.matched_center_error_samples_ms,
        grade_report.reference_count, grade_report.matched_count, false);
    write_metric("speech_region_start",
        grade_report.pause_alignment.speech_regions.matched_start_error_samples_ms,
        grade_report.pause_alignment.speech_regions.reference_count,
        grade_report.pause_alignment.speech_regions.matched_count, false);
    write_metric("speech_region_end",
        grade_report.pause_alignment.speech_regions.matched_end_error_samples_ms,
        grade_report.pause_alignment.speech_regions.reference_count,
        grade_report.pause_alignment.speech_regions.matched_count, false);
    write_metric("word_onset_start",
        grade_report.word_onset_alignment.matched_start_error_samples_ms,
        grade_report.word_onset_alignment.reference_count,
        grade_report.word_onset_alignment.matched_count, false);
    write_metric("intra_word_center",
        grade_report.intra_word_alignment.matched_center_error_samples_ms,
        grade_report.intra_word_alignment.reference_count,
        grade_report.intra_word_alignment.matched_count, false);
    write_metric("syllable_center", syllable_report.matched_error_samples_ms,
        syllable_report.target_count, syllable_report.matched_count, true);
    out << "}\n";
    return out.str();
}

static std::string predicted_neural_track_summary_json(
    const PredictedNeuralTrackResult& result,
    const GradeReport& grade_report)
{
    const double fallback_rate = result.accepted_count > 0
        ? static_cast<double>(result.deterministic_fallback_count) / result.accepted_count
        : 0.0;
    std::ostringstream out;
    out << std::fixed << std::setprecision(6)
        << "{\n"
        << "  \"prediction_count\": " << result.prediction_count << ",\n"
        << "  \"accepted_count\": " << result.accepted_count << ",\n"
        << "  \"deterministic_fallback_count\": " << result.deterministic_fallback_count << ",\n"
        << "  \"deterministic_fallback_rate\": " << fallback_rate << ",\n"
        << "  \"graded_matched_count\": " << grade_report.matched_count << ",\n"
        << "  \"graded_missing_count\": " << grade_report.missing_count << ",\n"
        << "  \"graded_extra_count\": " << grade_report.extra_count << ",\n"
        << "  \"graded_order_violations\": " << grade_report.order_violations << ",\n"
        << "  \"graded_mean_abs_center_error_ms\": "
        << grade_report.mean_abs_center_error_ms << "\n"
        << "}\n";
    return out.str();
}

static void append_monotonic_audio_features(
    std::ostringstream& out,
    const FOffgridAIStreamingAudioFeatureFrame& frame)
{
    out << std::log(std::max(frame.RMS, 1.0e-6f)) << ','
        << frame.RMSNorm << ',' << frame.DeltaRMS << ',' << frame.Flux << ',' << frame.ZCR << ','
        << frame.LowBandNorm << ',' << frame.MidBandNorm << ',' << frame.HighBandNorm << ','
        << frame.SpectralCentroidNorm << ',' << frame.Periodicity << ','
        << frame.RichLowBandNorm << ',' << frame.RichMidBandNorm << ',' << frame.RichHighBandNorm << ','
        << frame.RichSpectralCentroidNorm << ',' << frame.RichSpectralRolloffNorm << ','
        << frame.RichSpectralFlatness << ',' << frame.RichSpectralFlux << ','
        << frame.RichPeriodicity << ',' << frame.LearnedSpeechProbability;
}

static std::string monotonic_audio_frames_csv(
    const TArray<FOffgridAIStreamingAudioFeatureFrame>& frames)
{
    std::ostringstream out;
    out << "schema_version,frame_index,start_sec,end_sec,log_rms,rms_norm,delta_rms,flux,zcr,"
        << "low_band_norm,mid_band_norm,high_band_norm,spectral_centroid_norm,periodicity,"
        << "rich_low_band_norm,rich_mid_band_norm,rich_high_band_norm,rich_spectral_centroid_norm,"
        << "rich_spectral_rolloff_norm,rich_spectral_flatness,rich_spectral_flux,rich_periodicity,"
        << "teacher_speech_probability\n";
    out << std::fixed << std::setprecision(6);
    for (int32 index = 0; index < frames.Num(); ++index)
    {
        out << 1 << ',' << index << ',' << frames[index].AudioBufferStartSec << ','
            << frames[index].AudioBufferEndSec << ',';
        append_monotonic_audio_features(out, frames[index]);
        out << '\n';
    }
    return out.str();
}

static const HandmadeLabel* find_gold_label_for_planned_event(
    const FOffgridAITextVisemeEvent& event,
    const std::vector<HandmadeLabel>& labels)
{
    for (const HandmadeLabel& label : labels)
    {
        if (label.word_index == event.WordIndex
            && label.source_phone_word_index == event.SourcePhoneIndex
            && label.pose == to_std(event.PoseID))
        {
            return &label;
        }
    }
    return nullptr;
}

static std::string monotonic_tokens_csv(
    const FOffgridAITextVisemePlan& plan,
    const std::vector<HandmadeLabel>& labels,
    const std::vector<GoldWordTiming>& gold_words)
{
    struct TokenTarget {
        int32 EventIndex = INDEX_NONE;
        const HandmadeLabel* Gold = nullptr;
        std::string Pose;
        std::string Phone;
        int WordIndex = INDEX_NONE;
        int PhoneIndexInWord = INDEX_NONE;
        int PhoneGlobalIndex = INDEX_NONE;
        int SentenceIndex = INDEX_NONE;
        double DurationPrior = 0.075;
        double TextCenterNorm = 0.0;
        double Strength = 0.0;
        int VisualRole = 0;
        bool IsVowel = false;
        bool IsSilence = false;
        bool IsWordStart = false;
        bool IsWordEnd = false;
        bool IsRegionStart = false;
        bool IsRegionEnd = false;
        bool IsSentenceBoundary = false;
        double Center = -1.0;
        double Duration = 0.020;
    };
    std::vector<TokenTarget> targets;
    for (int32 event_index = 0; event_index < plan.Events.Num(); ++event_index)
    {
        const auto& event = plan.Events[event_index];
        if (!event.bIsRenderable) continue;
        TokenTarget target;
        target.EventIndex = event_index;
        target.Gold = find_gold_label_for_planned_event(event, labels);
        target.Pose = to_std(event.PoseID);
        target.Phone = to_std(event.SourcePhoneBase);
        target.WordIndex = event.WordIndex;
        target.PhoneIndexInWord = event.SourcePhoneIndex;
        target.PhoneGlobalIndex = event.SourcePhoneGlobalIndex;
        target.SentenceIndex = event.SentenceIndex;
        const FOffgridAIExpectedPhone* phone = plan.ExpectedPhones.IsValidIndex(event.SourcePhoneGlobalIndex)
            ? &plan.ExpectedPhones[event.SourcePhoneGlobalIndex] : nullptr;
        target.DurationPrior = phone ? phone->WeightSeconds : 0.075f;
        target.IsVowel = phone && phone->bIsVowel;
        target.VisualRole = static_cast<int>(event.VisualRole);
        target.TextCenterNorm = event.StartNorm + 0.5f * (event.EndNorm - event.StartNorm);
        target.Strength = event.Strength;
        if (target.Gold)
        {
            target.Center = 0.5 * (target.Gold->start + target.Gold->end);
            target.Duration = target.Gold->end - target.Gold->start;
        }
        targets.push_back(target);
    }
    for (size_t index = 0; index < targets.size(); ++index)
    {
        if (targets[index].Center >= 0.0) continue;
        size_t before = index;
        while (before > 0 && targets[before - 1].Center < 0.0) --before;
        const bool has_before = before > 0;
        size_t after = index + 1;
        while (after < targets.size() && targets[after].Center < 0.0) ++after;
        const bool has_after = after < targets.size();
        if (has_before && has_after)
        {
            const size_t before_index = before - 1;
            const double fraction = static_cast<double>(index - before_index)
                / static_cast<double>(after - before_index);
            targets[index].Center = targets[before_index].Center
                + fraction * (targets[after].Center - targets[before_index].Center);
        }
        else if (has_before)
            targets[index].Center = targets[before - 1].Center + 0.030 * (index - before + 1);
        else if (has_after)
            targets[index].Center = std::max(0.0, targets[after].Center - 0.030 * (after - index));
        else
            targets[index].Center = 0.030 * static_cast<double>(index);
    }

    auto gold_word = [&](int word_index) -> const GoldWordTiming* {
        const auto found = std::find_if(gold_words.begin(), gold_words.end(), [&](const auto& word) {
            return word.word_index == word_index;
        });
        return found == gold_words.end() ? nullptr : &*found;
    };
    for (size_t index = 0; index < targets.size(); ++index)
    {
        const int word_index = targets[index].WordIndex;
        const bool first_in_word = index == 0 || targets[index - 1].WordIndex != word_index;
        const bool last_in_word = index + 1 == targets.size() || targets[index + 1].WordIndex != word_index;
        targets[index].IsWordStart = first_in_word;
        targets[index].IsWordEnd = last_in_word;
        const GoldWordTiming* word = gold_word(word_index);
        const GoldWordTiming* previous = first_in_word && index > 0
            ? gold_word(targets[index - 1].WordIndex) : nullptr;
        const GoldWordTiming* next = last_in_word && index + 1 < targets.size()
            ? gold_word(targets[index + 1].WordIndex) : nullptr;
        targets[index].IsRegionStart = first_in_word
            && (!previous || !word || previous->speech_region_index != word->speech_region_index);
        targets[index].IsRegionEnd = last_in_word
            && (!next || !word || next->speech_region_index != word->speech_region_index);
    }

    auto silence_target = [](double center, double duration, double text_center) {
        TokenTarget silence;
        silence.Pose = "__SILENCE__";
        silence.Phone = "SIL";
        silence.IsSilence = true;
        silence.DurationPrior = 0.040;
        silence.Center = center;
        silence.Duration = std::max(0.010, duration);
        silence.TextCenterNorm = text_center;
        return silence;
    };
    std::vector<TokenTarget> sequence;
    if (!targets.empty())
    {
        const double first_start = targets.front().Center - 0.5 * targets.front().Duration;
        TokenTarget leading = silence_target(
            std::max(0.0, first_start - 0.050), 0.100, 0.0);
        leading.IsRegionStart = true;
        sequence.push_back(std::move(leading));
        for (size_t index = 0; index < targets.size(); ++index)
        {
            sequence.push_back(targets[index]);
            if (index + 1 >= targets.size()
                || targets[index].WordIndex == targets[index + 1].WordIndex) continue;
            const GoldWordTiming* before = gold_word(targets[index].WordIndex);
            const GoldWordTiming* after = gold_word(targets[index + 1].WordIndex);
            double gap_start = before ? before->end
                : targets[index].Center + 0.5 * targets[index].Duration;
            double gap_end = after ? after->start
                : targets[index + 1].Center - 0.5 * targets[index + 1].Duration;
            const double low = targets[index].Center + 0.005;
            const double high = targets[index + 1].Center - 0.005;
            double center = 0.5 * (gap_start + gap_end);
            if (high > low) center = std::clamp(center, low, high);
            else center = 0.5 * (targets[index].Center + targets[index + 1].Center);
            const double text_center = 0.5
                * (targets[index].TextCenterNorm + targets[index + 1].TextCenterNorm);
            TokenTarget silence = silence_target(center, gap_end - gap_start, text_center);
            silence.SentenceIndex = targets[index + 1].SentenceIndex;
            silence.IsSentenceBoundary =
                targets[index].SentenceIndex != targets[index + 1].SentenceIndex;
            if (silence.IsSentenceBoundary) silence.DurationPrior = 0.120;
            silence.IsRegionEnd = targets[index].IsRegionEnd;
            silence.IsRegionStart = targets[index + 1].IsRegionStart;
            sequence.push_back(std::move(silence));
        }
        const double last_end = targets.back().Center + 0.5 * targets.back().Duration;
        TokenTarget trailing = silence_target(last_end + 0.050, 0.100, 1.0);
        trailing.IsRegionEnd = true;
        sequence.push_back(std::move(trailing));
    }
    targets = std::move(sequence);

    std::ostringstream out;
    out << "schema_version,token_index,event_index,pose,phone_base,word_index,phone_index_in_word,"
        << "phone_global_index,sentence_index,duration_prior_sec,is_vowel,visual_role,text_center_norm,strength,"
        << "is_silence,is_word_start,is_word_end,is_region_start,is_region_end,is_sentence_boundary,"
        << "target_center_sec,target_duration_sec,has_mfa_target\n";
    out << std::fixed << std::setprecision(6);
    for (size_t token_index = 0; token_index < targets.size(); ++token_index)
    {
        const TokenTarget& target = targets[token_index];
        out << 1 << ',' << token_index << ',' << target.EventIndex << ','
            << csv_value(target.Pose) << ',' << csv_value(target.Phone) << ','
            << target.WordIndex << ',' << target.PhoneIndexInWord << ',' << target.PhoneGlobalIndex << ','
            << target.SentenceIndex << ','
            << target.DurationPrior << ',' << (target.IsVowel ? 1 : 0) << ','
            << target.VisualRole << ',' << target.TextCenterNorm << ',' << target.Strength << ','
            << (target.IsSilence ? 1 : 0) << ',' << (target.IsWordStart ? 1 : 0) << ','
            << (target.IsWordEnd ? 1 : 0) << ',' << (target.IsRegionStart ? 1 : 0) << ','
            << (target.IsRegionEnd ? 1 : 0) << ',' << (target.IsSentenceBoundary ? 1 : 0) << ','
            << target.Center << ',' << target.Duration << ',' << (target.Gold ? 1 : 0) << '\n';
    }
    return out.str();
}

static std::string classify_gap_family_by_duration(double gap_seconds)
{
    if (gap_seconds >= 0.180) return "hard_silence";
    if (gap_seconds >= 0.130) return "speech_region_gap";
    if (gap_seconds >= 0.065) return "word_gap";
    if (gap_seconds >= 0.020) return "micro_gap";
    return "continuous_speech";
}

static std::string gold_pause_family_at_time(
    double time_seconds,
    double audio_end_seconds,
    const std::vector<GoldWordTiming>& gold_words,
    const std::vector<GoldSpeechRegion>& gold_speech)
{
    for (size_t i = 1; i < gold_words.size(); ++i)
    {
        const auto& prev = gold_words[i - 1];
        const auto& next = gold_words[i];
        if (next.start <= prev.end)
        {
            continue;
        }
        if (time_seconds >= prev.end && time_seconds < next.start)
        {
            return classify_gap_family_by_duration(next.start - prev.end);
        }
    }

    for (const auto& region : gold_speech)
    {
        if (time_seconds >= region.start && time_seconds < region.end)
        {
            return "continuous_speech";
        }
    }

    double gap_start = 0.0;
    double gap_end = audio_end_seconds;
    bool found = false;
    for (size_t i = 0; i < gold_speech.size(); ++i)
    {
        const auto& region = gold_speech[i];
        if (time_seconds < region.start)
        {
            gap_end = region.start;
            if (i > 0)
            {
                gap_start = gold_speech[i - 1].end;
            }
            found = true;
            break;
        }
        gap_start = region.end;
    }
    if (!found && !gold_speech.empty())
    {
        gap_start = gold_speech.back().end;
    }
    return classify_gap_family_by_duration(std::max(0.0, gap_end - gap_start));
}

static bool is_pause_family(const std::string& family)
{
    return !family.empty() && family != "continuous_speech";
}

static std::vector<double> build_gold_pause_word_boundaries(
    const std::vector<GoldWordTiming>& gold_words,
    double min_gap_seconds)
{
    std::vector<double> boundaries;
    for (size_t i = 1; i < gold_words.size(); ++i)
    {
        const auto& prev = gold_words[i - 1];
        const auto& next = gold_words[i];
        if ((next.start - prev.end) >= min_gap_seconds)
        {
            boundaries.push_back(next.start);
        }
    }
    return boundaries;
}

static std::vector<double> build_predicted_pause_word_boundaries(const TArray<FOffgridAIStreamingAudioFeatureFrame>& frames)
{
    std::vector<double> boundaries;
    bool previous_pause = false;
    for (int32 i = 0; i < frames.Num(); ++i)
    {
        const std::string family = derived_pause_family(frames[i]);
        const bool pause = is_pause_family(family);
        if (previous_pause && !pause)
        {
            boundaries.push_back(static_cast<double>(frames[i].AudioBufferCenterSec));
        }
        previous_pause = pause;
    }
    return boundaries;
}

static std::vector<double> collect_match_errors_ms(
    int& out_matched,
    const std::vector<double>& gold_boundaries,
    const std::vector<double>& predicted_boundaries,
    double tolerance_seconds)
{
    std::vector<double> errors_ms;
    size_t predicted_index = 0;
    out_matched = 0;
    for (double gold_time : gold_boundaries)
    {
        while (predicted_index < predicted_boundaries.size() &&
               predicted_boundaries[predicted_index] < gold_time - tolerance_seconds)
        {
            ++predicted_index;
        }

        if (predicted_index >= predicted_boundaries.size())
        {
            continue;
        }

        size_t best_index = predicted_index;
        double best_error = std::abs(predicted_boundaries[best_index] - gold_time);
        if ((predicted_index + 1) < predicted_boundaries.size())
        {
            const double next_error = std::abs(predicted_boundaries[predicted_index + 1] - gold_time);
            if (next_error < best_error)
            {
                best_index = predicted_index + 1;
                best_error = next_error;
            }
        }

        if (best_error <= tolerance_seconds)
        {
            ++out_matched;
            errors_ms.push_back(best_error * 1000.0);
            predicted_index = best_index + 1;
        }
    }
    return errors_ms;
}

static std::string streaming_detector_frames_csv(
    StreamingDetectorReport& out_report,
    const TArray<FOffgridAIStreamingAudioFeatureFrame>& frames,
    const std::vector<GoldWordTiming>& gold_words,
    const std::vector<GoldSpeechRegion>& gold_speech,
    double audio_end_seconds)
{
    constexpr double kMinGoldGapSeconds = 0.020;
    constexpr double kBoundaryToleranceSeconds = 0.050;

    const std::vector<double> gold_boundaries = build_gold_pause_word_boundaries(gold_words, kMinGoldGapSeconds);
    const std::vector<double> predicted_boundaries = build_predicted_pause_word_boundaries(frames);
    out_report.gold_word_boundaries = static_cast<int>(gold_boundaries.size());
    out_report.predicted_word_boundaries = static_cast<int>(predicted_boundaries.size());
    std::vector<double> boundary_errors_ms = collect_match_errors_ms(
        out_report.matched_word_boundaries,
        gold_boundaries,
        predicted_boundaries,
        kBoundaryToleranceSeconds);
    std::sort(boundary_errors_ms.begin(), boundary_errors_ms.end());
    if (!boundary_errors_ms.empty())
    {
        const double sum = std::accumulate(boundary_errors_ms.begin(), boundary_errors_ms.end(), 0.0);
        out_report.mean_abs_word_boundary_error_ms = sum / static_cast<double>(boundary_errors_ms.size());
        out_report.median_abs_word_boundary_error_ms = percentile_ms(boundary_errors_ms, 0.5);
        out_report.p90_abs_word_boundary_error_ms = percentile_ms(boundary_errors_ms, 0.9);
    }

    std::ostringstream out;
    out << "frame_index,start,end,center,gold_pause_family,pred_pause_family,pause_confidence,pause_match,predicted_word_boundary\n";
    out << std::fixed << std::setprecision(6);

    bool previous_pause = false;
    for (int32 i = 0; i < frames.Num(); ++i)
    {
        const auto& frame = frames[i];
        const double center = static_cast<double>(frame.AudioBufferCenterSec);
        const std::string gold_pause = gold_pause_family_at_time(center, audio_end_seconds, gold_words, gold_speech);
        const std::string pred_pause = derived_pause_family(frame);
        const double pred_pause_confidence = derived_pause_confidence(frame);
        const bool pause = is_pause_family(pred_pause);
        const int predicted_word_boundary = (previous_pause && !pause) ? 1 : 0;
        previous_pause = pause;
        const int pause_match = (!gold_pause.empty() && gold_pause == pred_pause) ? 1 : 0;
        ++out_report.pause_frames;
        out_report.pause_matches += pause_match;

        out << i << ','
            << frame.AudioBufferStartSec << ','
            << frame.AudioBufferEndSec << ','
            << frame.AudioBufferCenterSec << ','
            << gold_pause << ','
            << pred_pause << ','
            << pred_pause_confidence << ','
            << pause_match << ','
            << predicted_word_boundary << '\n';
    }

    return out.str();
}

static std::vector<TimeSpan> build_predicted_speech_regions(const TArray<FOffgridAIStreamingSpeechRegion>& speech_regions)
{
    std::vector<TimeSpan> spans;
    for (const auto& speech_region : speech_regions)
    {
        const double start = static_cast<double>(speech_region.AudioBufferStartSec);
        const double end = static_cast<double>(std::max(speech_region.AudioBufferLastSpeechSec, speech_region.AudioBufferEndSec));
        if (end <= start)
        {
            continue;
        }
        spans.push_back(TimeSpan{speech_region.SpeechRegionIndex, start, end});
    }
    return spans;
}

static std::vector<TimeSpan> build_gold_speech_regions(const std::vector<GoldSpeechRegion>& regions)
{
    std::vector<TimeSpan> spans;
    for (const auto& region : regions)
    {
        spans.push_back(TimeSpan{region.index, region.start, region.end});
    }
    return spans;
}

static std::vector<TimeSpan> build_gold_sentence_regions(const std::vector<GoldWordTiming>& words)
{
    std::vector<TimeSpan> spans;
    for (const auto& word : words)
    {
        if (word.sentence_index < 0) continue;
        if (spans.empty() || spans.back().key != word.sentence_index)
        {
            spans.push_back(TimeSpan{word.sentence_index, word.start, word.end});
        }
        else
        {
            spans.back().end = std::max(spans.back().end, word.end);
        }
    }
    return spans;
}

static std::vector<TimeSpan> build_gold_clause_regions(const std::vector<GoldWordTiming>& words)
{
    std::vector<TimeSpan> spans;
    int last_sentence = std::numeric_limits<int>::min();
    int last_speech_region = std::numeric_limits<int>::min();
    int synthetic_key = -1;
    for (const auto& word : words)
    {
        if (word.speech_region_index < 0) continue;
        if (spans.empty() || word.sentence_index != last_sentence || word.speech_region_index != last_speech_region)
        {
            ++synthetic_key;
            spans.push_back(TimeSpan{synthetic_key, word.start, word.end});
            last_sentence = word.sentence_index;
            last_speech_region = word.speech_region_index;
        }
        else
        {
            spans.back().end = std::max(spans.back().end, word.end);
        }
    }
    return spans;
}

static std::vector<TimeSpan> build_predicted_sentence_regions(const FOffgridAIAlignedVisemeTrack& track)
{
    std::vector<TimeSpan> spans;
    for (const auto& event : track.Events)
    {
        if (!event.bIsRenderable || event.bCanceledByWordHandoff) continue;
        if (event.SentenceIndex < 0) continue;
        const double start = static_cast<double>(event.RenderStartSeconds);
        const double end = static_cast<double>(event.RenderEndSeconds);
        if (spans.empty() || spans.back().key != event.SentenceIndex)
        {
            spans.push_back(TimeSpan{event.SentenceIndex, start, end});
        }
        else
        {
            spans.back().end = std::max(spans.back().end, end);
        }
    }
    return spans;
}

static std::vector<TimeSpan> build_predicted_clause_regions(const FOffgridAIAlignedVisemeTrack& track)
{
    std::vector<TimeSpan> spans;
    int last_sentence = std::numeric_limits<int>::min();
    int last_speech_region = std::numeric_limits<int>::min();
    int synthetic_key = -1;
    for (const auto& event : track.Events)
    {
        if (!event.bIsRenderable || event.bCanceledByWordHandoff) continue;
        if (event.SpeechRegionIndex < 0) continue;
        const double start = static_cast<double>(event.RenderStartSeconds);
        const double end = static_cast<double>(event.RenderEndSeconds);
        if (spans.empty() || event.SentenceIndex != last_sentence || event.SpeechRegionIndex != last_speech_region)
        {
            ++synthetic_key;
            spans.push_back(TimeSpan{synthetic_key, start, end});
            last_sentence = event.SentenceIndex;
            last_speech_region = event.SpeechRegionIndex;
        }
        else
        {
            spans.back().end = std::max(spans.back().end, end);
        }
    }
    return spans;
}

static BoundaryAlignmentReport grade_region_boundaries(const std::vector<TimeSpan>& gold, const std::vector<TimeSpan>& predicted)
{
    BoundaryAlignmentReport report;
    report.reference_count = static_cast<int>(gold.size());
    report.predicted_count = static_cast<int>(predicted.size());
    report.count_mismatch = report.reference_count != report.predicted_count;

    if (gold.empty() || predicted.empty())
    {
        report.missing_count = report.reference_count;
        report.extra_count = report.predicted_count;
        return report;
    }

    const auto overlap_seconds = [](const TimeSpan& left, const TimeSpan& right) {
        return std::max(0.0, std::min(left.end, right.end) - std::max(left.start, right.start));
    };
    std::vector<int> gold_overlap_counts(gold.size(), 0);
    std::vector<int> predicted_overlap_counts(predicted.size(), 0);
    for (size_t gold_index = 0; gold_index < gold.size(); ++gold_index)
    {
        for (size_t predicted_index = 0; predicted_index < predicted.size(); ++predicted_index)
        {
            if (overlap_seconds(gold[gold_index], predicted[predicted_index]) <= 0.0) continue;
            ++gold_overlap_counts[gold_index];
            ++predicted_overlap_counts[predicted_index];
        }
    }
    report.merge_count = static_cast<int>(std::count_if(
        predicted_overlap_counts.begin(), predicted_overlap_counts.end(), [](int count) { return count > 1; }));
    report.split_count = static_cast<int>(std::count_if(
        gold_overlap_counts.begin(), gold_overlap_counts.end(), [](int count) { return count > 1; }));

    constexpr uint8_t kAlignmentNone = 0;
    constexpr uint8_t kAlignmentMatch = 1;
    constexpr uint8_t kAlignmentMissing = 2;
    constexpr uint8_t kAlignmentExtra = 3;
    struct AlignmentCell
    {
        double cost = std::numeric_limits<double>::infinity();
        uint8_t step = 0;
    };
    const size_t column_count = predicted.size() + 1;
    std::vector<AlignmentCell> cells((gold.size() + 1) * column_count);
    const auto cell_index = [column_count](size_t gold_index, size_t predicted_index) {
        return gold_index * column_count + predicted_index;
    };
    cells[0].cost = 0.0;
    for (size_t gold_index = 1; gold_index <= gold.size(); ++gold_index)
    {
        auto& cell = cells[cell_index(gold_index, 0)];
        cell.cost = static_cast<double>(gold_index);
        cell.step = kAlignmentMissing;
    }
    for (size_t predicted_index = 1; predicted_index <= predicted.size(); ++predicted_index)
    {
        auto& cell = cells[cell_index(0, predicted_index)];
        cell.cost = static_cast<double>(predicted_index);
        cell.step = kAlignmentExtra;
    }
    for (size_t gold_index = 1; gold_index <= gold.size(); ++gold_index)
    {
        for (size_t predicted_index = 1; predicted_index <= predicted.size(); ++predicted_index)
        {
            auto& cell = cells[cell_index(gold_index, predicted_index)];
            const double missing_cost = cells[cell_index(gold_index - 1, predicted_index)].cost + 1.0;
            const double extra_cost = cells[cell_index(gold_index, predicted_index - 1)].cost + 1.0;
            if (missing_cost <= extra_cost)
            {
                cell.cost = missing_cost;
                cell.step = kAlignmentMissing;
            }
            else
            {
                cell.cost = extra_cost;
                cell.step = kAlignmentExtra;
            }

            const auto& gold_span = gold[gold_index - 1];
            const auto& predicted_span = predicted[predicted_index - 1];
            const double overlap = overlap_seconds(gold_span, predicted_span);
            if (overlap <= 0.0) continue;
            const double span_union = std::max(gold_span.end, predicted_span.end)
                - std::min(gold_span.start, predicted_span.start);
            const double intersection_over_union = span_union > 0.0 ? overlap / span_union : 0.0;
            const double match_cost = cells[cell_index(gold_index - 1, predicted_index - 1)].cost
                + (1.0 - intersection_over_union);
            if (match_cost <= cell.cost)
            {
                cell.cost = match_cost;
                cell.step = kAlignmentMatch;
            }
        }
    }

    std::vector<std::pair<size_t, size_t>> matches;
    size_t gold_index = gold.size();
    size_t predicted_index = predicted.size();
    while (gold_index > 0 || predicted_index > 0)
    {
        const uint8_t step = cells[cell_index(gold_index, predicted_index)].step;
        if (step == kAlignmentMatch)
        {
            matches.emplace_back(gold_index - 1, predicted_index - 1);
            --gold_index;
            --predicted_index;
        }
        else if (step == kAlignmentMissing)
        {
            --gold_index;
        }
        else if (step == kAlignmentExtra)
        {
            --predicted_index;
        }
        else
        {
            break;
        }
    }
    std::reverse(matches.begin(), matches.end());
    report.matched_count = static_cast<int>(matches.size());
    report.missing_count = report.reference_count - report.matched_count;
    report.extra_count = report.predicted_count - report.matched_count;

    std::vector<double> start_errors;
    std::vector<double> end_errors;
    double lead_sum = 0.0;
    double tail_sum = 0.0;
    for (const auto& match : matches)
    {
        const auto& gold_span = gold[match.first];
        const auto& predicted_span = predicted[match.second];
        start_errors.push_back(std::abs(predicted_span.start - gold_span.start) * 1000.0);
        end_errors.push_back(std::abs(predicted_span.end - gold_span.end) * 1000.0);
        lead_sum += std::max(0.0, gold_span.start - predicted_span.start) * 1000.0;
        tail_sum += std::max(0.0, predicted_span.end - gold_span.end) * 1000.0;
        if (gold_overlap_counts[match.first] == 1 && predicted_overlap_counts[match.second] == 1)
        {
            ++report.clean_match_count;
        }
    }

    if (report.matched_count <= 0) return report;
    const double denom = static_cast<double>(report.matched_count);
    report.mean_abs_start_error_ms = mean_ms(start_errors);
    report.median_abs_start_error_ms = percentile_ms(start_errors, 0.50);
    report.p90_abs_start_error_ms = percentile_ms(start_errors, 0.90);
    report.mean_abs_end_error_ms = mean_ms(end_errors);
    report.median_abs_end_error_ms = percentile_ms(end_errors, 0.50);
    report.p90_abs_end_error_ms = percentile_ms(end_errors, 0.90);
    report.mean_lead_in_ms = lead_sum / denom;
    report.mean_tail_out_ms = tail_sum / denom;
    report.matched_start_error_samples_ms = start_errors;
    report.matched_end_error_samples_ms = end_errors;
    return report;
}

static WordOnsetAlignmentReport grade_word_onsets(
    const FOffgridAIAlignedVisemeTrack& track,
    const std::vector<HandmadeLabel>& handmade,
    const std::vector<GoldWordTiming>& gold_words)
{
    WordOnsetAlignmentReport report;
    std::vector<int> reference_word_indices;
    std::vector<int> seen_words;
    for (const auto& label : handmade)
    {
        if (label.word_index < 0) continue;
        if (std::find(seen_words.begin(), seen_words.end(), label.word_index) == seen_words.end())
        {
            seen_words.push_back(label.word_index);
            reference_word_indices.push_back(label.word_index);
        }
    }

    std::vector<double> errors;
    double early_sum = 0.0;
    double late_sum = 0.0;
    report.reference_count = static_cast<int>(reference_word_indices.size());
    for (int word_index : reference_word_indices)
    {
        const auto gold_it = std::find_if(gold_words.begin(), gold_words.end(), [&](const GoldWordTiming& word) {
            return word.word_index == word_index;
        });
        if (gold_it == gold_words.end())
        {
            continue;
        }

        const auto event_it = std::find_if(track.Events.begin(), track.Events.end(), [&](const FOffgridAIAlignedVisemeEvent& event) {
            return event.bIsRenderable && !event.bCanceledByWordHandoff
                && event.WordIndex == word_index;
        });
        if (event_it == track.Events.end())
        {
            continue;
        }

        ++report.matched_count;
        const double detected_start = static_cast<double>(event_it->RenderStartSeconds);
        const double error_ms = std::abs(detected_start - gold_it->start) * 1000.0;
        errors.push_back(error_ms);
        early_sum += std::max(0.0, gold_it->start - detected_start) * 1000.0;
        late_sum += std::max(0.0, detected_start - gold_it->start) * 1000.0;
    }

    report.missing_count = std::max(0, report.reference_count - report.matched_count);
    if (!errors.empty())
    {
        report.mean_abs_start_error_ms = mean_ms(errors);
        report.median_abs_start_error_ms = percentile_ms(errors, 0.50);
        report.p90_abs_start_error_ms = percentile_ms(errors, 0.90);
        report.mean_early_start_ms = early_sum / static_cast<double>(errors.size());
        report.mean_late_start_ms = late_sum / static_cast<double>(errors.size());
    }
    report.matched_start_error_samples_ms = errors;
    return report;
}

static IntraWordAlignmentReport grade_intra_word_alignment(
    const FOffgridAIAlignedVisemeTrack& track,
    const std::vector<HandmadeLabel>& handmade)
{
    IntraWordAlignmentReport report;
    std::vector<int> unique_word_indices;
    for (const auto& label : handmade)
    {
        if (label.word_index >= 0
            && std::find(unique_word_indices.begin(), unique_word_indices.end(), label.word_index) == unique_word_indices.end())
        {
            unique_word_indices.push_back(label.word_index);
        }
    }

    std::vector<double> center_errors;
    std::vector<double> start_errors;
    std::vector<double> end_errors;
    int matched_events_total = 0;
    int predicted_events_total = 0;

    for (int word_index : unique_word_indices)
    {
        std::vector<int> label_indices;
        for (size_t i = 0; i < handmade.size(); ++i)
        {
            if (handmade[i].word_index == word_index)
            {
                label_indices.push_back(static_cast<int>(i));
            }
        }
        std::vector<int> event_indices;
        for (int32 i = 0; i < track.Events.Num(); ++i)
        {
            if (track.Events[i].bIsRenderable
                && !track.Events[i].bCanceledByWordHandoff
                && track.Events[i].WordIndex == word_index)
            {
                event_indices.push_back(i);
            }
        }
        if (label_indices.size() <= 1)
        {
            continue;
        }

        label_indices.erase(label_indices.begin());
        report.reference_count += static_cast<int>(label_indices.size());
        if (!event_indices.empty())
        {
            event_indices.erase(event_indices.begin());
        }
        predicted_events_total += static_cast<int>(event_indices.size());

        const std::vector<GradeMatch> matches = compute_monotonic_matches_subset(track, handmade, label_indices, event_indices);
        matched_events_total += static_cast<int>(matches.size());
        for (const GradeMatch& match : matches)
        {
            const HandmadeLabel& label = handmade[static_cast<size_t>(match.label_index)];
            const auto& event = track.Events[match.event_index];
            const double center = (label.start + label.end) * 0.5;
            center_errors.push_back(std::abs(static_cast<double>(event.FinalRenderCenterSeconds) - center) * 1000.0);
            start_errors.push_back(std::abs(static_cast<double>(event.RenderStartSeconds) - label.start) * 1000.0);
            end_errors.push_back(std::abs(static_cast<double>(event.RenderEndSeconds) - label.end) * 1000.0);
        }
    }

    report.matched_count = matched_events_total;
    report.missing_count = std::max(0, report.reference_count - report.matched_count);
    report.extra_count = std::max(0, predicted_events_total - report.matched_count);
    if (!center_errors.empty())
    {
        report.mean_abs_center_error_ms = mean_ms(center_errors);
        report.median_abs_center_error_ms = percentile_ms(center_errors, 0.50);
        report.p90_abs_center_error_ms = percentile_ms(center_errors, 0.90);
        report.mean_abs_start_error_ms = mean_ms(start_errors);
        report.mean_abs_end_error_ms = mean_ms(end_errors);
    }
    report.matched_center_error_samples_ms = center_errors;
    report.matched_start_error_samples_ms = start_errors;
    return report;
}

static SpeechRegionContainmentReport grade_speech_region_containment(
    const FOffgridAIAlignedVisemeTrack& track,
    const TArray<FOffgridAIStreamingSpeechRegion>& speech_regions)
{
    SpeechRegionContainmentReport report;
    report.predicted_region_count = speech_regions.Num();
    report.event_count = 0;

    std::vector<double> early_leaks_ms;
    std::vector<double> late_leaks_ms;
    for (const auto& event : track.Events)
    {
        if (!event.bIsRenderable || event.bCanceledByWordHandoff) continue;
        ++report.event_count;
        if (!speech_regions.IsValidIndex(event.SpeechRegionIndex))
        {
            ++report.events_with_invalid_region;
            continue;
        }

        const auto& region = speech_regions[event.SpeechRegionIndex];
        const double region_start = static_cast<double>(region.AudioBufferStartSec);
        const double region_end = static_cast<double>(std::max(region.AudioBufferLastSpeechSec, region.AudioBufferEndSec));
        const double event_start = static_cast<double>(event.RenderStartSeconds);
        const double event_end = static_cast<double>(event.RenderEndSeconds);
        const double early_ms = std::max(0.0, region_start - event_start) * 1000.0;
        const double late_ms = std::max(0.0, event_end - region_end) * 1000.0;
        if (early_ms > 0.0 || late_ms > 0.0)
        {
            ++report.events_outside_region;
        }
        if (early_ms > 0.0)
        {
            ++report.events_starting_before_region;
            early_leaks_ms.push_back(early_ms);
        }
        if (late_ms > 0.0)
        {
            ++report.events_ending_after_region;
            late_leaks_ms.push_back(late_ms);
        }
    }

    report.mean_early_start_leak_ms = mean_ms(early_leaks_ms);
    report.mean_late_end_leak_ms = mean_ms(late_leaks_ms);
    report.max_early_start_leak_ms = early_leaks_ms.empty() ? 0.0 : *std::max_element(early_leaks_ms.begin(), early_leaks_ms.end());
    report.max_late_end_leak_ms = late_leaks_ms.empty() ? 0.0 : *std::max_element(late_leaks_ms.begin(), late_leaks_ms.end());
    return report;
}

static GradeReport grade(
    const FOffgridAIAlignedVisemeTrack& track,
    const TArray<FOffgridAIStreamingSpeechRegion>& speech_regions,
    const std::vector<HandmadeLabel>& handmade,
    const std::vector<GoldWordTiming>& gold_words,
    const std::vector<GoldSpeechRegion>& gold_speech)
{
    GradeReport report;
    report.reference_count = static_cast<int>(handmade.size());
    report.committed_count = 0;
    for (const auto& event : track.Events)
    {
        if (event.bIsRenderable && !event.bCanceledByWordHandoff)
            ++report.committed_count;
    }
    report.dropped_visemes.dropped_count = 0;
    report.dropped_visemes.dropped_region_closed_count = 0;
    report.dropped_visemes.dropped_missing_region_count = 0;
    report.dropped_visemes.dropped_strong_visible_count = 0;
    report.dropped_visemes.dropped_rate = 0.0;

    std::vector<double> errors;
    std::vector<double> start_errors;
    std::vector<double> end_errors;
    std::vector<double> duration_errors;
    for (int32 i = 1; i < track.Events.Num(); ++i)
    {
        if (track.Events[i].EventIndex <= track.Events[i - 1].EventIndex
            || track.Events[i].FinalRenderCenterSeconds + 1e-6f < track.Events[i - 1].FinalRenderCenterSeconds)
        {
            ++report.order_violations;
        }
    }

    const std::vector<GradeMatch> matches = compute_monotonic_matches(track, handmade);
    report.matched_count = static_cast<int>(matches.size());
    for (const GradeMatch& match : matches)
    {
        const HandmadeLabel& label = handmade[static_cast<size_t>(match.label_index)];
        const auto& event = track.Events[match.event_index];
        const double center = (label.start + label.end) * 0.5;
        const double err_ms = std::abs(static_cast<double>(event.FinalRenderCenterSeconds) - center) * 1000.0;
        errors.push_back(err_ms);
        start_errors.push_back(std::abs(static_cast<double>(event.RenderStartSeconds) - label.start) * 1000.0);
        end_errors.push_back(std::abs(static_cast<double>(event.RenderEndSeconds) - label.end) * 1000.0);
        duration_errors.push_back(std::abs(
            (static_cast<double>(event.RenderEndSeconds) - static_cast<double>(event.RenderStartSeconds))
            - (label.end - label.start)) * 1000.0);
        if (event.RenderStartSeconds + 0.050f < label.start) report.early_start_rate += 1.0;
        if (event.RenderEndSeconds - 0.050f > label.end) report.late_tail_rate += 1.0;
    }

    report.missing_count = report.reference_count - report.matched_count;
    report.extra_count = report.committed_count - report.matched_count;
    if (!errors.empty())
    {
        report.mean_abs_center_error_ms = mean_ms(errors);
        report.median_abs_center_error_ms = percentile_ms(errors, 0.50);
        report.p90_abs_center_error_ms = percentile_ms(errors, 0.90);
        report.mean_abs_start_error_ms = mean_ms(start_errors);
        report.mean_abs_end_error_ms = mean_ms(end_errors);
        report.mean_abs_duration_error_ms = mean_ms(duration_errors);
        report.early_start_rate /= static_cast<double>(errors.size());
        report.late_tail_rate /= static_cast<double>(errors.size());
    }

    report.pause_alignment.speech_regions = grade_region_boundaries(
        build_gold_speech_regions(gold_speech),
        build_predicted_speech_regions(speech_regions));
    report.pause_alignment.text_sentence_spans = grade_region_boundaries(
        build_gold_sentence_regions(gold_words),
        build_predicted_sentence_regions(track));
    report.pause_alignment.text_sentence_region_subspans = grade_region_boundaries(
        build_gold_clause_regions(gold_words),
        build_predicted_clause_regions(track));
    report.word_onset_alignment = grade_word_onsets(track, handmade, gold_words);
    report.intra_word_alignment = grade_intra_word_alignment(track, handmade);
    report.speech_region_containment = grade_speech_region_containment(track, speech_regions);
    report.matched_center_error_samples_ms = errors;
    report.matched_start_error_samples_ms = start_errors;
    return report;
}

static std::string word_onset_diagnostics_csv(
    const FOffgridAIAlignedVisemeTrack& track,
    const std::vector<HandmadeLabel>& handmade,
    const std::vector<GoldWordTiming>& gold_words)
{
    std::ostringstream out;
    out << "word_index,word,gold_word_start,gold_word_end,first_gold_pose,first_gold_start,first_event_pose,first_event_start,detected_word_start,detected_word_start_mapped,error_ms,missing_event\n";
    std::vector<int> seen_words;
    for (const auto& label : handmade)
    {
        if (label.word_index < 0) continue;
        if (std::find(seen_words.begin(), seen_words.end(), label.word_index) != seen_words.end()) continue;
        seen_words.push_back(label.word_index);

        const auto gold_it = std::find_if(gold_words.begin(), gold_words.end(), [&](const GoldWordTiming& word) {
            return word.word_index == label.word_index;
        });
        const auto event_it = std::find_if(track.Events.begin(), track.Events.end(), [&](const FOffgridAIAlignedVisemeEvent& event) {
            return event.bIsRenderable && !event.bCanceledByWordHandoff
                && event.WordIndex == label.word_index;
        });

        const double gold_start = gold_it != gold_words.end() ? gold_it->start : label.start;
        const double gold_end = gold_it != gold_words.end() ? gold_it->end : label.end;
        out << label.word_index << "," << (gold_it != gold_words.end() ? gold_it->word : label.word) << ","
            << gold_start << "," << gold_end << ","
            << label.pose << "," << label.start << ",";
        if (event_it == track.Events.end())
        {
            out << ",,,,0,true\n";
        }
        else
        {
            const double event_start = static_cast<double>(event_it->RenderStartSeconds);
            out << to_std(event_it->PoseID) << "," << event_start << ","
                << event_start << ","
                << 0 << ","
                << std::abs(event_start - gold_start) * 1000.0 << ",false\n";
        }
    }
    return out.str();
}

static std::string focus_alignment_grade_json(
    const FOffgridAIAlignedVisemeTrack& track,
    const FOffgridAITextVisemePlan& plan,
    const std::vector<HandmadeLabel>& handmade,
    const std::vector<GoldSpeechRegion>& gold_speech)
{
    struct CenterRow
    {
        int word_index = -1;
        int region_index = -1;
        std::string word;
        std::string phone;
        double gold_center = 0.0;
        double runtime_center = 0.0;
    };

    std::vector<CenterRow> words;
    std::vector<int> expected_word_indices;
    std::vector<int> expected_region_indices;
    for (const auto& label : handmade)
    {
        if (label.word_index < 0) continue;
        if (std::find(expected_word_indices.begin(), expected_word_indices.end(), label.word_index)
            == expected_word_indices.end())
            expected_word_indices.push_back(label.word_index);
        if (label.speech_region_index >= 0
            && std::find(expected_region_indices.begin(), expected_region_indices.end(), label.speech_region_index)
                == expected_region_indices.end())
            expected_region_indices.push_back(label.speech_region_index);
        if (std::any_of(words.begin(), words.end(), [&](const CenterRow& row) {
            return row.word_index == label.word_index;
        })) continue;
        const auto event = std::find_if(track.Events.begin(), track.Events.end(), [&](const auto& candidate) {
            return candidate.bIsRenderable && !candidate.bCanceledByWordHandoff
                && candidate.WordIndex == label.word_index
                && (label.source_phone_global_index < 0
                    || candidate.SourcePhoneIndex == label.source_phone_global_index);
        });
        if (event == track.Events.end()) continue;
        words.push_back({
            label.word_index,
            label.speech_region_index,
            label.word,
            label.source_phone_base,
            0.5 * (label.start + label.end),
            event->FinalRenderCenterSeconds });
    }

    std::vector<CenterRow> regions;
    for (const auto& word : words)
    {
        if (std::any_of(regions.begin(), regions.end(), [&](const CenterRow& row) {
            return row.region_index == word.region_index;
        })) continue;
        regions.push_back(word);
    }

    // Priority-0 perceptual target: the first animated pose in a speech
    // region should peak on the first syllable nucleus, even when that pose
    // depicts a consonant onset (for example W in "what").
    auto is_vowel_phone = [](const std::string& phone) {
        static const std::vector<std::string> vowels = {
            "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER",
            "EY", "IH", "IY", "OW", "OY", "UH", "UW"
        };
        return std::find(vowels.begin(), vowels.end(), phone) != vowels.end();
    };
    std::vector<CenterRow> region_nuclei;
    for (const auto& region : regions)
    {
        const auto nucleus = std::find_if(handmade.begin(), handmade.end(), [&](const HandmadeLabel& label) {
            return label.word_index == region.word_index
                && label.speech_region_index == region.region_index
                && is_vowel_phone(label.source_phone_base);
        });
        const auto first_event = std::find_if(track.Events.begin(), track.Events.end(), [&](const auto& event) {
            return event.bIsRenderable && !event.bCanceledByWordHandoff
                && event.WordIndex == region.word_index;
        });
        if (nucleus == handmade.end() || first_event == track.Events.end()) continue;
        region_nuclei.push_back({
            region.word_index,
            region.region_index,
            nucleus->word,
            nucleus->source_phone_base,
            0.5 * (nucleus->start + nucleus->end),
            first_event->FinalRenderCenterSeconds });
    }

    auto mean_abs_ms = [](const std::vector<CenterRow>& rows) {
        double total = 0.0;
        for (const auto& row : rows)
            total += std::abs(row.runtime_center - row.gold_center) * 1000.0;
        return rows.empty() ? 0.0 : total / static_cast<double>(rows.size());
    };
    // A resumed region is owned incorrectly if its first viseme center lands
    // before the gold speech start, inside the preceding nonspeech interval.
    auto region_resume_in_pause_ms = [&](const CenterRow& row) {
        const auto region = std::find_if(gold_speech.begin(), gold_speech.end(), [&](const GoldSpeechRegion& candidate) {
            return candidate.index == row.region_index;
        });
        if (region == gold_speech.end() || region == gold_speech.begin()) return 0.0;
        return std::max(0.0, region->start - row.runtime_center) * 1000.0;
    };
    auto is_region_resume = [&](const CenterRow& row) {
        const auto region = std::find_if(gold_speech.begin(), gold_speech.end(), [&](const GoldSpeechRegion& candidate) {
            return candidate.index == row.region_index;
        });
        return region != gold_speech.end() && region != gold_speech.begin();
    };
    int expected_region_resume_count = 0;
    int region_resume_in_pause_count = 0;
    double region_resume_in_pause_max_ms = 0.0;
    for (const auto& row : regions)
    {
        const auto region = std::find_if(gold_speech.begin(), gold_speech.end(), [&](const GoldSpeechRegion& candidate) {
            return candidate.index == row.region_index;
        });
        if (region == gold_speech.end() || region == gold_speech.begin()) continue;
        ++expected_region_resume_count;
        const double lead_ms = region_resume_in_pause_ms(row);
        if (lead_ms > 0.0) ++region_resume_in_pause_count;
        region_resume_in_pause_max_ms = std::max(region_resume_in_pause_max_ms, lead_ms);
    }
    // Subtract the region-head error so this measures cursor drift after the
    // region was anchored, rather than counting a uniformly shifted region.
    double within_region_early_drift_max_ms = 0.0;
    for (const auto& word : words)
    {
        const auto region = std::find_if(regions.begin(), regions.end(), [&](const CenterRow& candidate) {
            return candidate.region_index == word.region_index;
        });
        if (region == regions.end()) continue;
        const double region_error = region->runtime_center - region->gold_center;
        const double word_error = word.runtime_center - word.gold_center;
        within_region_early_drift_max_ms = std::max(
            within_region_early_drift_max_ms,
            std::max(0.0, region_error - word_error) * 1000.0);
    }
    struct PauseBoundaryRow
    {
        int region_before = -1;
        int region_after = -1;
        double start = 0.0;
        double end = 0.0;
        int samples = 0;
        int active_samples = 0;
    };
    int pause_samples = 0;
    int active_pause_samples = 0;
    std::vector<PauseBoundaryRow> pause_boundaries;
    for (size_t index = 0; index + 1 < gold_speech.size(); ++index)
    {
        PauseBoundaryRow boundary;
        boundary.region_before = gold_speech[index].index;
        boundary.region_after = gold_speech[index + 1].index;
        boundary.start = gold_speech[index].end;
        boundary.end = gold_speech[index + 1].start;
        for (double seconds = gold_speech[index].end;
             seconds + 0.0005 < gold_speech[index + 1].start;
             seconds += 0.001)
        {
            ++pause_samples;
            ++boundary.samples;
            if (FOffgridAIVisemePerformer::Sample(
                    track, static_cast<float>(seconds), true).Num() > 0)
            {
                ++active_pause_samples;
                ++boundary.active_samples;
            }
        }
        pause_boundaries.push_back(boundary);
    }

    std::ostringstream out;
    const int planned_event_count = plan.Events.Num();
    const int committed_event_count = track.Events.Num();
    const double event_completion_rate = planned_event_count > 0
        ? static_cast<double>(committed_event_count) / static_cast<double>(planned_event_count)
        : 1.0;
    const double word_head_coverage_rate = !expected_word_indices.empty()
        ? static_cast<double>(words.size()) / static_cast<double>(expected_word_indices.size())
        : 1.0;
    const double region_head_coverage_rate = !expected_region_indices.empty()
        ? static_cast<double>(regions.size()) / static_cast<double>(expected_region_indices.size())
        : 1.0;
    const double pause_clean_rate = pause_samples > 0
        ? 1.0 - static_cast<double>(active_pause_samples) / static_cast<double>(pause_samples)
        : 1.0;
    out << std::fixed << std::setprecision(3);
    out << "{\n"
        << "  \"planned_event_count\": " << planned_event_count << ",\n"
        << "  \"committed_event_count\": " << committed_event_count << ",\n"
        << "  \"event_completion_rate\": " << event_completion_rate << ",\n"
        << "  \"expected_region_head_count\": " << expected_region_indices.size() << ",\n"
        << "  \"matched_region_head_count\": " << regions.size() << ",\n"
        << "  \"region_head_coverage_rate\": " << region_head_coverage_rate << ",\n"
        << "  \"region_head_mean_abs_ms\": " << mean_abs_ms(regions) << ",\n"
        << "  \"expected_region_nucleus_count\": " << expected_region_indices.size() << ",\n"
        << "  \"matched_region_nucleus_count\": " << region_nuclei.size() << ",\n"
        << "  \"region_nucleus_mean_abs_ms\": " << mean_abs_ms(region_nuclei) << ",\n"
        << "  \"expected_region_resume_count\": " << expected_region_resume_count << ",\n"
        << "  \"region_resume_in_pause_count\": " << region_resume_in_pause_count << ",\n"
        << "  \"region_resume_in_pause_rate\": "
        << (expected_region_resume_count > 0
            ? static_cast<double>(region_resume_in_pause_count) / static_cast<double>(expected_region_resume_count)
            : 0.0) << ",\n"
        << "  \"region_resume_in_pause_max_ms\": " << region_resume_in_pause_max_ms << ",\n"
        << "  \"expected_word_head_count\": " << expected_word_indices.size() << ",\n"
        << "  \"matched_word_head_count\": " << words.size() << ",\n"
        << "  \"word_head_coverage_rate\": " << word_head_coverage_rate << ",\n"
        << "  \"word_head_mean_abs_ms\": " << mean_abs_ms(words) << ",\n"
        << "  \"within_region_early_drift_max_ms\": " << within_region_early_drift_max_ms << ",\n"
        << "  \"pause_duration_ms\": " << pause_samples << ",\n"
        << "  \"pause_animation_leakage_ms\": " << active_pause_samples << ",\n"
        << "  \"pause_clean_rate\": " << pause_clean_rate << ",\n"
        << "  \"pause_boundaries\": [\n";
    for (size_t index = 0; index < pause_boundaries.size(); ++index)
    {
        const auto& boundary = pause_boundaries[index];
        out << "    {\"region_before\": " << boundary.region_before
            << ", \"region_after\": " << boundary.region_after
            << ", \"start\": " << boundary.start
            << ", \"end\": " << boundary.end
            << ", \"duration_ms\": " << boundary.samples
            << ", \"animation_leakage_ms\": " << boundary.active_samples
            << ", \"clean_rate\": "
            << (boundary.samples > 0
                ? 1.0 - static_cast<double>(boundary.active_samples)
                    / static_cast<double>(boundary.samples)
                : 1.0)
            << "}"
            << (index + 1 < pause_boundaries.size() ? "," : "") << "\n";
    }
    out << "  ],\n"
        << "  \"region_heads\": [\n";
    for (size_t index = 0; index < regions.size(); ++index)
    {
        const auto& row = regions[index];
        out << "    {\"region_index\": " << row.region_index
            << ", \"word_index\": " << row.word_index
            << ", \"word\": \"" << row.word << "\", \"phone\": \"" << row.phone
            << "\", \"gold_center\": " << row.gold_center
            << ", \"runtime_center\": " << row.runtime_center
            << ", \"error_ms\": " << (row.runtime_center - row.gold_center) * 1000.0
            << ", \"is_resume\": " << (is_region_resume(row) ? "true" : "false")
            << ", \"resume_in_pause_ms\": " << region_resume_in_pause_ms(row) << "}"
            << (index + 1 < regions.size() ? "," : "") << "\n";
    }
    out << "  ],\n  \"region_nuclei\": [\n";
    for (size_t index = 0; index < region_nuclei.size(); ++index)
    {
        const auto& row = region_nuclei[index];
        out << "    {\"region_index\": " << row.region_index
            << ", \"word_index\": " << row.word_index
            << ", \"word\": \"" << row.word << "\", \"nucleus_phone\": \"" << row.phone
            << "\", \"gold_nucleus_center\": " << row.gold_center
            << ", \"first_viseme_center\": " << row.runtime_center
            << ", \"error_ms\": " << (row.runtime_center - row.gold_center) * 1000.0 << "}"
            << (index + 1 < region_nuclei.size() ? "," : "") << "\n";
    }
    out << "  ],\n  \"word_heads\": [\n";
    for (size_t index = 0; index < words.size(); ++index)
    {
        const auto& row = words[index];
        const auto region = std::find_if(regions.begin(), regions.end(), [&](const CenterRow& candidate) {
            return candidate.region_index == row.region_index;
        });
        double relative_early_drift_ms = 0.0;
        if (region != regions.end())
        {
            const double region_error = region->runtime_center - region->gold_center;
            const double word_error = row.runtime_center - row.gold_center;
            relative_early_drift_ms = std::max(0.0, region_error - word_error) * 1000.0;
        }
        out << "    {\"word_index\": " << row.word_index
            << ", \"word\": \"" << row.word << "\", \"phone\": \"" << row.phone
            << "\", \"gold_center\": " << row.gold_center
            << ", \"runtime_center\": " << row.runtime_center
            << ", \"error_ms\": " << (row.runtime_center - row.gold_center) * 1000.0
            << ", \"relative_early_drift_ms\": " << relative_early_drift_ms << "}"
            << (index + 1 < words.size() ? "," : "") << "\n";
    }
    out << "  ]\n}\n";
    return out.str();
}

static std::array<double, 6> pose_channels(const FOffgridAILipsyncPoseRuntimeState& pose)
{
    return {
        pose.Open,
        pose.Closed,
        pose.Wide,
        pose.Round,
        pose.Funnel,
        pose.Teeth
    };
}

static std::string runtime_delivery_grade_json(
    const FOffgridAIAlignedVisemeTrack& track,
    const FOffgridAITextVisemePlan& plan,
    const std::vector<GoldWordTiming>& gold_words,
    const std::vector<GoldSpeechRegion>& gold_speech)
{
    constexpr double sample_seconds = 0.010;
    constexpr double visible_weight = 0.20;
    constexpr int minimum_visible_samples = 2;

    struct Delivery
    {
        double peak = 0.0;
        int visible_samples = 0;
    };
    std::map<int, Delivery> delivery;
    double end_seconds = std::max(0.0, static_cast<double>(track.SpeechEndSeconds));
    for (const auto& event : track.Events)
        end_seconds = std::max(end_seconds, static_cast<double>(event.RenderEndSeconds));
    for (double seconds = 0.0; seconds <= end_seconds + 0.0001; seconds += sample_seconds)
    {
        const auto samples = FOffgridAIVisemePerformer::Sample(
            track, static_cast<float>(seconds), true);
        for (const auto& sample : samples)
        {
            if (sample.EventIndex < 0 || sample.SubmittedWeight < visible_weight) continue;
            const auto event = std::find_if(
                track.Events.begin(), track.Events.end(), [&](const auto& candidate) {
                    return candidate.EventIndex == sample.EventIndex;
                });
            if (event == track.Events.end()
                || seconds + 0.0005 < static_cast<double>(event->CommitPlaybackSeconds))
                continue;
            Delivery& row = delivery[sample.EventIndex];
            row.peak = std::max(row.peak, static_cast<double>(sample.SubmittedWeight));
            ++row.visible_samples;
        }
    }

    auto is_planned_candidate = [](const FOffgridAITextVisemeEvent& event) {
        return event.bIsRenderable && event.Strength > 0.012f;
    };
    auto is_committed_candidate = [](const FOffgridAIAlignedVisemeEvent& event) {
        return event.bIsRenderable && !event.bCanceledByWordHandoff
            && event.Strength > 0.012f;
    };
    auto was_delivered = [&](const auto& event) {
        const auto found = delivery.find(event.EventIndex);
        return found != delivery.end()
            && found->second.peak >= visible_weight
            && found->second.visible_samples >= minimum_visible_samples;
    };

    int planned_candidates = 0;
    std::map<int, int> planned_by_word;
    std::map<int, int> sentence_for_word;
    for (const auto& event : plan.Events)
    {
        if (!is_planned_candidate(event)) continue;
        ++planned_candidates;
        ++planned_by_word[event.WordIndex];
        sentence_for_word[event.WordIndex] = event.SentenceIndex;
    }

    int committed_candidates = 0;
    int delivered_events = 0;
    int late_after_window = 0;
    std::map<int, int> delivered_by_word;
    std::map<int, std::vector<const FOffgridAIAlignedVisemeEvent*>> committed_by_word;
    for (const auto& event : track.Events)
    {
        if (!is_committed_candidate(event)) continue;
        ++committed_candidates;
        committed_by_word[event.WordIndex].push_back(&event);
        if (event.CommitPlaybackSeconds > event.RenderEndSeconds + 0.001f)
            ++late_after_window;
        if (was_delivered(event))
        {
            ++delivered_events;
            ++delivered_by_word[event.WordIndex];
        }
    }

    int expected_words = 0;
    int missing_words = 0;
    int compressed_words = 0;
    std::vector<double> word_duration_ratios;
    for (const auto& gold : gold_words)
    {
        if (planned_by_word[gold.word_index] <= 0) continue;
        ++expected_words;
        if (delivered_by_word[gold.word_index] <= 0) ++missing_words;
        const auto found = committed_by_word.find(gold.word_index);
        if (found == committed_by_word.end() || found->second.size() < 2) continue;
        double runtime_start = std::numeric_limits<double>::infinity();
        double runtime_end = 0.0;
        for (const auto* event : found->second)
        {
            runtime_start = std::min(runtime_start, static_cast<double>(event->RenderStartSeconds));
            runtime_end = std::max(runtime_end, static_cast<double>(event->RenderEndSeconds));
        }
        const double gold_duration = std::max(0.0, gold.end - gold.start);
        const double runtime_duration = std::max(0.0, runtime_end - runtime_start);
        if (gold_duration >= 0.120)
            word_duration_ratios.push_back(runtime_duration / gold_duration);
        if (gold_duration >= 0.120
            && runtime_duration + 0.080 < gold_duration
            && runtime_duration < 0.45 * gold_duration)
            ++compressed_words;
    }

    int empty_speech_regions = 0;
    for (const auto& region : gold_speech)
    {
        bool has_delivery = false;
        for (const auto& event : track.Events)
        {
            if (!is_committed_candidate(event) || !was_delivered(event)) continue;
            if (event.FinalRenderCenterSeconds >= region.start - 0.001
                && event.FinalRenderCenterSeconds <= region.end + 0.001)
            {
                has_delivery = true;
                break;
            }
        }
        if (!has_delivery) ++empty_speech_regions;
    }

    struct SentenceWindow
    {
        double gold_start = std::numeric_limits<double>::infinity();
        double gold_end = 0.0;
        int expected_events = 0;
        std::vector<const FOffgridAIAlignedVisemeEvent*> events;
    };
    std::map<int, SentenceWindow> sentences;
    for (const auto& event : plan.Events)
        if (is_planned_candidate(event)) ++sentences[event.SentenceIndex].expected_events;
    for (const auto& gold : gold_words)
    {
        const auto found = sentence_for_word.find(gold.word_index);
        if (found == sentence_for_word.end()) continue;
        auto& sentence = sentences[found->second];
        sentence.gold_start = std::min(sentence.gold_start, gold.start);
        sentence.gold_end = std::max(sentence.gold_end, gold.end);
    }
    for (const auto& event : track.Events)
        if (is_committed_candidate(event)) sentences[event.SentenceIndex].events.push_back(&event);

    int expected_sentences = 0;
    int missing_sentences = 0;
    int compressed_sentences = 0;
    std::vector<double> sentence_duration_ratios;
    for (const auto& pair : sentences)
    {
        const SentenceWindow& sentence = pair.second;
        if (sentence.expected_events <= 0 || !std::isfinite(sentence.gold_start)) continue;
        ++expected_sentences;
        bool delivered_in_window = false;
        double runtime_start = std::numeric_limits<double>::infinity();
        double runtime_end = 0.0;
        for (const auto* event : sentence.events)
        {
            runtime_start = std::min(runtime_start, static_cast<double>(event->RenderStartSeconds));
            runtime_end = std::max(runtime_end, static_cast<double>(event->RenderEndSeconds));
            if (was_delivered(*event)
                && event->FinalRenderCenterSeconds >= sentence.gold_start - 0.001
                && event->FinalRenderCenterSeconds <= sentence.gold_end + 0.001)
                delivered_in_window = true;
        }
        if (!delivered_in_window) ++missing_sentences;
        const double gold_duration = sentence.gold_end - sentence.gold_start;
        const double runtime_duration = std::max(0.0, runtime_end - runtime_start);
        if (sentence.events.size() >= 2 && gold_duration >= 0.300)
            sentence_duration_ratios.push_back(runtime_duration / gold_duration);
        if (sentence.events.size() >= 2 && gold_duration >= 0.300
            && runtime_duration + 0.150 < gold_duration
            && runtime_duration < 0.65 * gold_duration)
            ++compressed_sentences;
    }

    const auto rate = [](int numerator, int denominator) {
        return denominator > 0
            ? static_cast<double>(numerator) / static_cast<double>(denominator)
            : 1.0;
    };
    std::ostringstream out;
    out << std::fixed << std::setprecision(6)
        << "{\n"
        << "  \"planned_candidate_events\": " << planned_candidates << ",\n"
        << "  \"committed_candidate_events\": " << committed_candidates << ",\n"
        << "  \"delivered_events\": " << delivered_events << ",\n"
        << "  \"event_delivery_rate\": " << rate(delivered_events, planned_candidates) << ",\n"
        << "  \"late_after_window_events\": " << late_after_window << ",\n"
        << "  \"expected_words\": " << expected_words << ",\n"
        << "  \"missing_words\": " << missing_words << ",\n"
        << "  \"word_delivery_rate\": " << rate(expected_words - missing_words, expected_words) << ",\n"
        << "  \"compressed_words\": " << compressed_words << ",\n"
        << "  \"word_duration_ratio_mean\": " << mean_ms(word_duration_ratios) << ",\n"
        << "  \"word_duration_ratio_median\": " << percentile_ms(word_duration_ratios, 0.5) << ",\n"
        << "  \"word_duration_ratios\": [";
    for (size_t index = 0; index < word_duration_ratios.size(); ++index)
    {
        if (index > 0) out << ',';
        out << word_duration_ratios[index];
    }
    out << "],\n"
        << "  \"expected_speech_regions\": " << gold_speech.size() << ",\n"
        << "  \"empty_speech_regions\": " << empty_speech_regions << ",\n"
        << "  \"speech_region_delivery_rate\": "
        << rate(static_cast<int>(gold_speech.size()) - empty_speech_regions,
            static_cast<int>(gold_speech.size())) << ",\n"
        << "  \"expected_sentences\": " << expected_sentences << ",\n"
        << "  \"missing_sentences\": " << missing_sentences << ",\n"
        << "  \"compressed_sentences\": " << compressed_sentences << ",\n"
        << "  \"sentence_duration_ratio_mean\": " << mean_ms(sentence_duration_ratios) << ",\n"
        << "  \"sentence_duration_ratio_median\": " << percentile_ms(sentence_duration_ratios, 0.5) << ",\n"
        << "  \"sentence_duration_ratios\": [";
    for (size_t index = 0; index < sentence_duration_ratios.size(); ++index)
    {
        if (index > 0) out << ',';
        out << sentence_duration_ratios[index];
    }
    out << "],\n"
        << "  \"sentence_delivery_rate\": "
        << rate(expected_sentences - missing_sentences, expected_sentences) << "\n"
        << "}\n";
    return out.str();
}

static std::string presentation_motion_grade_json(
    const FOffgridAIAlignedVisemeTrack& track)
{
    constexpr double frame_seconds = 1.0 / 60.0;
    double end_seconds = std::max(0.0, static_cast<double>(track.SpeechEndSeconds));
    for (const auto& event : track.Events)
    {
        end_seconds = std::max(
            end_seconds,
            static_cast<double>(event.FinalRenderCenterSeconds) + 0.300);
    }

    FOffgridAILipsyncPoseRuntimeState displayed;
    std::array<double, 6> previous_channels{};
    std::array<double, 6> previous_velocity{};
    double squared_step_sum = 0.0;
    double squared_acceleration_sum = 0.0;
    double max_channel_step = 0.0;
    double max_channel_speed = 0.0;
    double max_channel_acceleration = 0.0;
    std::map<std::string, double> previous_driver_weights;
    std::map<std::string, double> previous_driver_velocity;
    double driver_squared_step_sum = 0.0;
    double driver_squared_acceleration_sum = 0.0;
    double driver_max_pose_step = 0.0;
    double driver_max_pose_speed = 0.0;
    double driver_max_pose_acceleration = 0.0;
    int driver_channel_sample_count = 0;
    int sample_count = 0;
    int channel_sample_count = 0;
    double jaw_peak = 0.0;
    double jaw_sum = 0.0;
    double jaw_squared_speed_sum = 0.0;
    double jaw_max_speed = 0.0;
    double previous_jaw_speed = 0.0;
    int jaw_high_aperture_samples = 0;
    int jaw_velocity_reversals = 0;
    int jaw_nucleus_samples = 0;
    double jaw_nucleus_abs_error_sum = 0.0;

    for (double seconds = 0.0; seconds <= end_seconds + 0.0001;
         seconds += frame_seconds)
    {
        const auto samples = FOffgridAIVisemePerformer::Sample(
            track, static_cast<float>(seconds), true);
        const auto weights = FOffgridAIVisemePerformer::CollapseByPoseID(samples);
        std::map<std::string, double> driver_weights;
        for (const auto& event : track.Events)
        {
            if (event.PoseID.IsNone()) continue;
            driver_weights[to_std(event.PoseID)] = weights.FindRef(event.PoseID);
        }
        driver_weights["JawOpen"] = weights.FindRef(TEXT("JawOpen"));
        if (sample_count > 0)
        {
            std::map<std::string, bool> pose_ids;
            for (const auto& pair : previous_driver_weights) pose_ids[pair.first] = true;
            for (const auto& pair : driver_weights) pose_ids[pair.first] = true;
            for (const auto& pose : pose_ids)
            {
                const double previous = previous_driver_weights.count(pose.first)
                    ? previous_driver_weights.at(pose.first) : 0.0;
                const double current = driver_weights.count(pose.first)
                    ? driver_weights.at(pose.first) : 0.0;
                const double step = current - previous;
                const double velocity = step / frame_seconds;
                const double old_velocity = previous_driver_velocity.count(pose.first)
                    ? previous_driver_velocity.at(pose.first) : 0.0;
                const double acceleration =
                    (velocity - old_velocity) / frame_seconds;
                driver_max_pose_step = std::max(
                    driver_max_pose_step, std::abs(step));
                driver_max_pose_speed = std::max(
                    driver_max_pose_speed, std::abs(velocity));
                driver_max_pose_acceleration = std::max(
                    driver_max_pose_acceleration, std::abs(acceleration));
                driver_squared_step_sum += step * step;
                driver_squared_acceleration_sum += acceleration * acceleration;
                previous_driver_velocity[pose.first] = velocity;
                ++driver_channel_sample_count;
            }
        }
        previous_driver_weights = driver_weights;
        const auto target =
            FOffgridAIVisemePerformer::BuildPoseStateFromPoseWeights(weights);
        displayed = FOffgridAIVisemePerformer::StepDisplayedPose(
            displayed,
            target,
            static_cast<float>(frame_seconds),
            0.035f,
            false);
        const auto channels = pose_channels(displayed);
        const double jaw = displayed.Open;
        jaw_peak = std::max(jaw_peak, jaw);
        jaw_sum += jaw;
        if (jaw >= 0.75) ++jaw_high_aperture_samples;
        if (sample_count > 0)
        {
            const double jaw_speed = (jaw - previous_channels[0]) / frame_seconds;
            jaw_max_speed = std::max(jaw_max_speed, std::abs(jaw_speed));
            jaw_squared_speed_sum += jaw_speed * jaw_speed;
            if (std::abs(jaw_speed) >= 0.05
                && std::abs(previous_jaw_speed) >= 0.05
                && jaw_speed * previous_jaw_speed < 0.0)
            {
                ++jaw_velocity_reversals;
            }
            previous_jaw_speed = jaw_speed;
        }
        for (const auto& event : track.Events)
        {
            if (event.JawOpenTarget < 0.0f
                || event.bCanceledByWordHandoff
                || std::abs(seconds - event.FinalRenderCenterSeconds)
                    > frame_seconds * 0.5)
                continue;
            jaw_nucleus_abs_error_sum += std::abs(
                jaw - static_cast<double>(event.JawOpenTarget));
            ++jaw_nucleus_samples;
        }
        if (sample_count > 0)
        {
            for (size_t channel = 0; channel < channels.size(); ++channel)
            {
                const double step = channels[channel] - previous_channels[channel];
                const double velocity = step / frame_seconds;
                const double acceleration =
                    (velocity - previous_velocity[channel]) / frame_seconds;
                max_channel_step = std::max(max_channel_step, std::abs(step));
                max_channel_speed = std::max(max_channel_speed, std::abs(velocity));
                max_channel_acceleration = std::max(
                    max_channel_acceleration, std::abs(acceleration));
                squared_step_sum += step * step;
                squared_acceleration_sum += acceleration * acceleration;
                previous_velocity[channel] = velocity;
                ++channel_sample_count;
            }
        }
        previous_channels = channels;
        ++sample_count;
    }

    const double rms_channel_step = channel_sample_count > 0
        ? std::sqrt(squared_step_sum / static_cast<double>(channel_sample_count))
        : 0.0;
    const double rms_channel_acceleration = channel_sample_count > 0
        ? std::sqrt(squared_acceleration_sum / static_cast<double>(channel_sample_count))
        : 0.0;
    const double driver_rms_pose_step = driver_channel_sample_count > 0
        ? std::sqrt(driver_squared_step_sum
            / static_cast<double>(driver_channel_sample_count))
        : 0.0;
    const double driver_rms_pose_acceleration = driver_channel_sample_count > 0
        ? std::sqrt(driver_squared_acceleration_sum
            / static_cast<double>(driver_channel_sample_count))
        : 0.0;
    const double jaw_mean = sample_count > 0
        ? jaw_sum / static_cast<double>(sample_count)
        : 0.0;
    const double jaw_rms_speed = sample_count > 1
        ? std::sqrt(jaw_squared_speed_sum
            / static_cast<double>(sample_count - 1))
        : 0.0;
    const double jaw_high_aperture_fraction = sample_count > 0
        ? static_cast<double>(jaw_high_aperture_samples)
            / static_cast<double>(sample_count)
        : 0.0;
    const double jaw_nucleus_mean_abs_error = jaw_nucleus_samples > 0
        ? jaw_nucleus_abs_error_sum
            / static_cast<double>(jaw_nucleus_samples)
        : 0.0;
    std::ostringstream out;
    out << std::fixed << std::setprecision(6)
        << "{\n"
        << "  \"sample_rate_hz\": 60.0,\n"
        << "  \"sample_count\": " << sample_count << ",\n"
        << "  \"max_channel_step_per_frame\": " << max_channel_step << ",\n"
        << "  \"rms_channel_step_per_frame\": " << rms_channel_step << ",\n"
        << "  \"max_channel_speed_per_second\": " << max_channel_speed << ",\n"
        << "  \"max_channel_acceleration_per_second2\": "
        << max_channel_acceleration << ",\n"
        << "  \"rms_channel_acceleration_per_second2\": "
        << rms_channel_acceleration << ",\n"
        << "  \"driver_max_pose_step_per_frame\": "
        << driver_max_pose_step << ",\n"
        << "  \"driver_rms_pose_step_per_frame\": "
        << driver_rms_pose_step << ",\n"
        << "  \"driver_max_pose_speed_per_second\": "
        << driver_max_pose_speed << ",\n"
        << "  \"driver_max_pose_acceleration_per_second2\": "
        << driver_max_pose_acceleration << ",\n"
        << "  \"driver_rms_pose_acceleration_per_second2\": "
        << driver_rms_pose_acceleration << ",\n"
        << "  \"jaw_peak_aperture\": " << jaw_peak << ",\n"
        << "  \"jaw_mean_aperture\": " << jaw_mean << ",\n"
        << "  \"jaw_high_aperture_fraction\": "
        << jaw_high_aperture_fraction << ",\n"
        << "  \"jaw_max_speed_per_second\": " << jaw_max_speed << ",\n"
        << "  \"jaw_rms_speed_per_second\": " << jaw_rms_speed << ",\n"
        << "  \"jaw_velocity_reversals\": " << jaw_velocity_reversals << ",\n"
        << "  \"jaw_nucleus_sample_count\": " << jaw_nucleus_samples << ",\n"
        << "  \"jaw_nucleus_mean_abs_target_error\": "
        << jaw_nucleus_mean_abs_error << "\n"
        << "}\n";
    return out.str();
}

static std::string region_ownership_grade_json(
    const FOffgridAIAlignedVisemeTrack& track,
    const FOffgridAITextVisemePlan& plan,
    const std::vector<GoldWordTiming>& gold_words,
    const std::vector<GoldSpeechRegion>& gold_speech)
{
    struct WordRow
    {
        int word_index = -1;
        std::string word;
        int transcript_region = -1;
        int mfa_region = -1;
        int runtime_region = -1;
        int runtime_mapped_mfa_region = -1;
        int planned_events = 0;
        int committed_events = 0;
        std::vector<int> runtime_regions;
        std::vector<int> runtime_mapped_mfa_regions;
        bool runtime_region_integrity = false;
        bool runtime_mfa_first_event_correct = false;
        bool runtime_mfa_majority_correct = false;
        bool runtime_mfa_strict_correct = false;
    };

    auto map_runtime_region_to_mfa = [&](int runtime_region_index) {
        const auto runtime = std::find_if(
            track.SpeechRegions.begin(), track.SpeechRegions.end(), [&](const auto& candidate) {
                return candidate.SpeechRegionIndex == runtime_region_index;
            });
        if (runtime == track.SpeechRegions.end() || gold_speech.empty()) return -1;

        const double runtime_start = runtime->StartSeconds;
        const double runtime_end = runtime->EndSeconds;
        int best_region = -1;
        double best_overlap = 0.0;
        for (const auto& gold : gold_speech)
        {
            const double overlap = std::max(
                0.0, std::min(runtime_end, gold.end) - std::max(runtime_start, gold.start));
            if (overlap > best_overlap)
            {
                best_overlap = overlap;
                best_region = gold.index;
            }
        }
        if (best_region >= 0) return best_region;

        const double midpoint = 0.5 * (runtime_start + runtime_end);
        double best_distance = std::numeric_limits<double>::max();
        for (const auto& gold : gold_speech)
        {
            const double distance = midpoint < gold.start
                ? gold.start - midpoint
                : (midpoint > gold.end ? midpoint - gold.end : 0.0);
            if (distance < best_distance)
            {
                best_distance = distance;
                best_region = gold.index;
            }
        }
        return best_region;
    };

    auto append_unique = [](std::vector<int>& values, int value) {
        if (value >= 0 && std::find(values.begin(), values.end(), value) == values.end())
            values.push_back(value);
    };
    auto mode_for_word = [](const FOffgridAIAlignedVisemeTrack& candidate_track, int word_index) {
        std::map<int, int> counts;
        for (const auto& event : candidate_track.Events)
            if (event.WordIndex == word_index && event.SpeechRegionIndex >= 0)
                ++counts[event.SpeechRegionIndex];
        int mode = -1;
        int best_count = 0;
        for (const auto& pair : counts)
            if (pair.second > best_count)
            {
                mode = pair.first;
                best_count = pair.second;
            }
        return mode;
    };

    std::vector<WordRow> words;
    int planned_word_count = 0;
    int committed_word_count = 0;
    int intact_word_count = 0;
    int strict_correct_count = 0;
    int majority_correct_count = 0;
    int transcript_index_correct_count = 0;
    int planned_event_count = 0;
    int committed_event_count = 0;
    int fully_committed_word_count = 0;
    int fully_committed_and_strict_correct_word_count = 0;
    for (const auto& gold_word : gold_words)
    {
        WordRow row;
        row.word_index = gold_word.word_index;
        row.word = gold_word.word;
        row.mfa_region = gold_word.speech_region_index;
        for (const auto& event : plan.Events)
        {
            if (event.WordIndex != row.word_index) continue;
            ++row.planned_events;
            if (row.transcript_region < 0) row.transcript_region = event.SpeechRegionIndex;
        }
        for (const auto& event : track.Events)
        {
            if (event.WordIndex != row.word_index) continue;
            ++row.committed_events;
            append_unique(row.runtime_regions, event.SpeechRegionIndex);
            append_unique(
                row.runtime_mapped_mfa_regions,
                map_runtime_region_to_mfa(event.SpeechRegionIndex));
        }
        row.runtime_region = mode_for_word(track, row.word_index);
        row.runtime_mapped_mfa_region = map_runtime_region_to_mfa(row.runtime_region);
        const int sole_runtime_region = row.runtime_regions.size() == 1
            ? row.runtime_regions.front()
            : -1;
        const int events_in_sole_runtime_region = static_cast<int>(std::count_if(
            track.Events.begin(), track.Events.end(), [&](const auto& event) {
                return event.WordIndex == row.word_index
                    && event.SpeechRegionIndex == sole_runtime_region;
            }));
        row.runtime_region_integrity = row.committed_events > 0
            && sole_runtime_region >= 0
            && events_in_sole_runtime_region == row.committed_events;
        const auto first_runtime_event = std::find_if(
            track.Events.begin(), track.Events.end(), [&](const auto& event) {
                return event.WordIndex == row.word_index;
            });
        row.runtime_mfa_first_event_correct = first_runtime_event != track.Events.end()
            && map_runtime_region_to_mfa(first_runtime_event->SpeechRegionIndex) == row.mfa_region;
        row.runtime_mfa_majority_correct = row.committed_events > 0
            && row.runtime_mapped_mfa_region == row.mfa_region;
        row.runtime_mfa_strict_correct = row.runtime_region_integrity
            && !row.runtime_mapped_mfa_regions.empty()
            && std::all_of(
                row.runtime_mapped_mfa_regions.begin(),
                row.runtime_mapped_mfa_regions.end(),
                [&](int region) { return region == row.mfa_region; });
        if (row.planned_events > 0) ++planned_word_count;
        if (row.committed_events > 0) ++committed_word_count;
        if (row.runtime_region_integrity) ++intact_word_count;
        if (row.runtime_mfa_strict_correct) ++strict_correct_count;
        if (row.runtime_mfa_majority_correct) ++majority_correct_count;
        if (row.transcript_region >= 0 && row.transcript_region == row.mfa_region)
            ++transcript_index_correct_count;
        planned_event_count += row.planned_events;
        committed_event_count += row.committed_events;
        if (row.planned_events > 0 && row.committed_events == row.planned_events)
        {
            ++fully_committed_word_count;
            if (row.runtime_mfa_strict_correct)
                ++fully_committed_and_strict_correct_word_count;
        }
        words.push_back(row);
    }

    struct BoundaryCounts
    {
        int compared = 0;
        int agreements = 0;
        int reference_splits = 0;
        int predicted_splits = 0;
        int true_splits = 0;
    };
    auto add_boundary = [](BoundaryCounts& counts, bool reference_split, bool predicted_split) {
        ++counts.compared;
        if (reference_split == predicted_split) ++counts.agreements;
        if (reference_split) ++counts.reference_splits;
        if (predicted_split) ++counts.predicted_splits;
        if (reference_split && predicted_split) ++counts.true_splits;
    };
    BoundaryCounts transcript_vs_mfa;
    BoundaryCounts runtime_vs_mfa;
    BoundaryCounts runtime_vs_transcript;
    for (size_t index = 1; index < words.size(); ++index)
    {
        const auto& before = words[index - 1];
        const auto& after = words[index];
        const bool mfa_split = before.mfa_region >= 0 && after.mfa_region >= 0
            && before.mfa_region != after.mfa_region;
        if (before.transcript_region >= 0 && after.transcript_region >= 0)
        {
            const bool transcript_split = before.transcript_region != after.transcript_region;
            add_boundary(transcript_vs_mfa, mfa_split, transcript_split);
            if (before.runtime_region >= 0 && after.runtime_region >= 0)
            {
                const bool runtime_split = before.runtime_region != after.runtime_region;
                add_boundary(runtime_vs_transcript, transcript_split, runtime_split);
            }
        }
        if (before.runtime_region >= 0 && after.runtime_region >= 0)
        {
            const bool runtime_split = before.runtime_region != after.runtime_region;
            add_boundary(runtime_vs_mfa, mfa_split, runtime_split);
        }
    }

    auto rate = [](int numerator, int denominator, double fallback = 1.0) {
        return denominator > 0
            ? static_cast<double>(numerator) / static_cast<double>(denominator)
            : fallback;
    };
    auto write_boundary_metrics = [&](std::ostringstream& out, const char* prefix, const BoundaryCounts& counts) {
        const double precision = rate(counts.true_splits, counts.predicted_splits, 1.0);
        const double recall = rate(counts.true_splits, counts.reference_splits, 1.0);
        const double f1 = precision + recall > 0.0
            ? 2.0 * precision * recall / (precision + recall)
            : 0.0;
        out << "  \"" << prefix << "_boundary_count\": " << counts.compared << ",\n"
            << "  \"" << prefix << "_reference_split_count\": " << counts.reference_splits << ",\n"
            << "  \"" << prefix << "_predicted_split_count\": " << counts.predicted_splits << ",\n"
            << "  \"" << prefix << "_true_split_count\": " << counts.true_splits << ",\n"
            << "  \"" << prefix << "_boundary_agreement_rate\": "
            << rate(counts.agreements, counts.compared) << ",\n"
            << "  \"" << prefix << "_split_precision\": " << precision << ",\n"
            << "  \"" << prefix << "_split_recall\": " << recall << ",\n"
            << "  \"" << prefix << "_split_f1\": " << f1 << ",\n";
    };

    std::ostringstream out;
    out << std::fixed << std::setprecision(6);
    out << "{\n"
        << "  \"lipsync_implementation_version\": \""
        << FOffgridAILipsyncRuntimeSession::GetImplementationVersion() << "\",\n"
        << "  \"lipsync_diagnostic_schema_version\": "
        << FOffgridAILipsyncRuntimeSession::GetDiagnosticSchemaVersion() << ",\n"
        << "  \"mfa_word_count\": " << words.size() << ",\n"
        << "  \"transcript_planned_word_count\": " << planned_word_count << ",\n"
        << "  \"runtime_committed_word_count\": " << committed_word_count << ",\n"
        << "  \"runtime_word_coverage_rate\": " << rate(committed_word_count, words.size()) << ",\n"
        << "  \"planned_event_count\": " << planned_event_count << ",\n"
        << "  \"committed_event_count\": " << committed_event_count << ",\n"
        << "  \"runtime_event_completion_rate\": " << rate(committed_event_count, planned_event_count) << ",\n"
        << "  \"runtime_fully_committed_word_rate\": " << rate(fully_committed_word_count, words.size()) << ",\n"
        << "  \"runtime_fully_committed_and_strict_correct_word_rate\": "
        << rate(fully_committed_and_strict_correct_word_count, words.size()) << ",\n"
        << "  \"runtime_word_region_integrity_rate\": " << rate(intact_word_count, committed_word_count) << ",\n"
        << "  \"runtime_mfa_word_region_strict_accuracy\": " << rate(strict_correct_count, committed_word_count) << ",\n"
        << "  \"runtime_mfa_word_region_majority_accuracy\": " << rate(majority_correct_count, committed_word_count) << ",\n"
        << "  \"transcript_mfa_raw_region_index_accuracy\": " << rate(transcript_index_correct_count, planned_word_count) << ",\n";
    write_boundary_metrics(out, "transcript_mfa", transcript_vs_mfa);
    write_boundary_metrics(out, "runtime_mfa", runtime_vs_mfa);
    write_boundary_metrics(out, "runtime_transcript", runtime_vs_transcript);
    out << "  \"words\": [\n";
    for (size_t index = 0; index < words.size(); ++index)
    {
        const auto& row = words[index];
        out << "    {\"word_index\": " << row.word_index
            << ", \"word\": \"" << row.word << "\""
            << ", \"transcript_region_index\": " << row.transcript_region
            << ", \"mfa_region_index\": " << row.mfa_region
            << ", \"runtime_region_index\": " << row.runtime_region
            << ", \"runtime_mapped_mfa_region_index\": " << row.runtime_mapped_mfa_region
            << ", \"planned_event_count\": " << row.planned_events
            << ", \"committed_event_count\": " << row.committed_events
            << ", \"runtime_regions\": [";
        for (size_t region_index = 0; region_index < row.runtime_regions.size(); ++region_index)
            out << (region_index > 0 ? ", " : "") << row.runtime_regions[region_index];
        out << "], \"runtime_mapped_mfa_regions\": [";
        for (size_t region_index = 0; region_index < row.runtime_mapped_mfa_regions.size(); ++region_index)
            out << (region_index > 0 ? ", " : "") << row.runtime_mapped_mfa_regions[region_index];
        out << "], \"runtime_region_integrity\": " << (row.runtime_region_integrity ? "true" : "false")
            << ", \"runtime_mfa_first_event_correct\": " << (row.runtime_mfa_first_event_correct ? "true" : "false")
            << ", \"runtime_mfa_majority_correct\": " << (row.runtime_mfa_majority_correct ? "true" : "false")
            << ", \"runtime_mfa_strict_correct\": " << (row.runtime_mfa_strict_correct ? "true" : "false")
            << "}" << (index + 1 < words.size() ? "," : "") << "\n";
    }
    out << "  ]\n}\n";
    return out.str();
}

static std::string grade_json(const GradeReport& grade_report)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(3);
    out << "{\n"
        << "  \"lipsync_implementation_version\": \""
        << FOffgridAILipsyncRuntimeSession::GetImplementationVersion() << "\",\n"
        << "  \"gold_available\": " << (grade_report.gold_available ? "true" : "false") << ",\n"
        << "  \"gold_status\": \"" << grade_report.gold_status << "\",\n"
        << "  \"reference_count\": " << grade_report.reference_count << ",\n"
        << "  \"committed_count\": " << grade_report.committed_count << ",\n"
        << "  \"matched_count\": " << grade_report.matched_count << ",\n"
        << "  \"missing_count\": " << grade_report.missing_count << ",\n"
        << "  \"extra_count\": " << grade_report.extra_count << ",\n"
        << "  \"order_violations\": " << grade_report.order_violations << ",\n"
        << "  \"dropped_visemes\": {\n"
        << "    \"dropped_count\": " << grade_report.dropped_visemes.dropped_count << ",\n"
        << "    \"dropped_region_closed_count\": " << grade_report.dropped_visemes.dropped_region_closed_count << ",\n"
        << "    \"dropped_missing_region_count\": " << grade_report.dropped_visemes.dropped_missing_region_count << ",\n"
        << "    \"dropped_strong_visible_count\": " << grade_report.dropped_visemes.dropped_strong_visible_count << ",\n"
        << "    \"dropped_rate\": " << grade_report.dropped_visemes.dropped_rate << "\n"
        << "  },\n"
        << "  \"mean_abs_center_error_ms\": " << grade_report.mean_abs_center_error_ms << ",\n"
        << "  \"median_abs_center_error_ms\": " << grade_report.median_abs_center_error_ms << ",\n"
        << "  \"p90_abs_center_error_ms\": " << grade_report.p90_abs_center_error_ms << ",\n"
        << "  \"mean_abs_start_error_ms\": " << grade_report.mean_abs_start_error_ms << ",\n"
        << "  \"mean_abs_end_error_ms\": " << grade_report.mean_abs_end_error_ms << ",\n"
        << "  \"mean_abs_duration_error_ms\": " << grade_report.mean_abs_duration_error_ms << ",\n"
        << "  \"early_start_rate\": " << grade_report.early_start_rate << ",\n"
        << "  \"late_tail_rate\": " << grade_report.late_tail_rate << ",\n"
        << "  \"pause_alignment\": {\n"
        << "    \"speech_region_reference_count\": " << grade_report.pause_alignment.speech_regions.reference_count << ",\n"
        << "    \"speech_region_predicted_count\": " << grade_report.pause_alignment.speech_regions.predicted_count << ",\n"
        << "    \"speech_region_matched_count\": " << grade_report.pause_alignment.speech_regions.matched_count << ",\n"
        << "    \"speech_region_missing_count\": " << grade_report.pause_alignment.speech_regions.missing_count << ",\n"
        << "    \"speech_region_extra_count\": " << grade_report.pause_alignment.speech_regions.extra_count << ",\n"
        << "    \"speech_region_clean_match_count\": " << grade_report.pause_alignment.speech_regions.clean_match_count << ",\n"
        << "    \"speech_region_merge_count\": " << grade_report.pause_alignment.speech_regions.merge_count << ",\n"
        << "    \"speech_region_split_count\": " << grade_report.pause_alignment.speech_regions.split_count << ",\n"
        << "    \"mean_abs_speech_region_start_error_ms\": " << grade_report.pause_alignment.speech_regions.mean_abs_start_error_ms << ",\n"
        << "    \"median_abs_speech_region_start_error_ms\": " << grade_report.pause_alignment.speech_regions.median_abs_start_error_ms << ",\n"
        << "    \"p90_abs_speech_region_start_error_ms\": " << grade_report.pause_alignment.speech_regions.p90_abs_start_error_ms << ",\n"
        << "    \"mean_abs_speech_region_end_error_ms\": " << grade_report.pause_alignment.speech_regions.mean_abs_end_error_ms << ",\n"
        << "    \"median_abs_speech_region_end_error_ms\": " << grade_report.pause_alignment.speech_regions.median_abs_end_error_ms << ",\n"
        << "    \"p90_abs_speech_region_end_error_ms\": " << grade_report.pause_alignment.speech_regions.p90_abs_end_error_ms << ",\n"
        << "    \"speech_region_count_mismatch\": " << (grade_report.pause_alignment.speech_regions.count_mismatch ? "true" : "false") << ",\n"
        << "    \"mean_speech_region_lead_in_ms\": " << grade_report.pause_alignment.speech_regions.mean_lead_in_ms << ",\n"
        << "    \"mean_speech_region_tail_out_ms\": " << grade_report.pause_alignment.speech_regions.mean_tail_out_ms << ",\n"
        << "    \"text_sentence_span_reference_count\": " << grade_report.pause_alignment.text_sentence_spans.reference_count << ",\n"
        << "    \"text_sentence_span_predicted_count\": " << grade_report.pause_alignment.text_sentence_spans.predicted_count << ",\n"
        << "    \"text_sentence_span_matched_count\": " << grade_report.pause_alignment.text_sentence_spans.matched_count << ",\n"
        << "    \"text_sentence_span_missing_count\": " << grade_report.pause_alignment.text_sentence_spans.missing_count << ",\n"
        << "    \"text_sentence_span_extra_count\": " << grade_report.pause_alignment.text_sentence_spans.extra_count << ",\n"
        << "    \"mean_abs_text_sentence_span_start_error_ms\": " << grade_report.pause_alignment.text_sentence_spans.mean_abs_start_error_ms << ",\n"
        << "    \"median_abs_text_sentence_span_start_error_ms\": " << grade_report.pause_alignment.text_sentence_spans.median_abs_start_error_ms << ",\n"
        << "    \"p90_abs_text_sentence_span_start_error_ms\": " << grade_report.pause_alignment.text_sentence_spans.p90_abs_start_error_ms << ",\n"
        << "    \"mean_abs_text_sentence_span_end_error_ms\": " << grade_report.pause_alignment.text_sentence_spans.mean_abs_end_error_ms << ",\n"
        << "    \"median_abs_text_sentence_span_end_error_ms\": " << grade_report.pause_alignment.text_sentence_spans.median_abs_end_error_ms << ",\n"
        << "    \"p90_abs_text_sentence_span_end_error_ms\": " << grade_report.pause_alignment.text_sentence_spans.p90_abs_end_error_ms << ",\n"
        << "    \"text_sentence_span_count_mismatch\": " << (grade_report.pause_alignment.text_sentence_spans.count_mismatch ? "true" : "false") << ",\n"
        << "    \"mean_text_sentence_span_lead_in_ms\": " << grade_report.pause_alignment.text_sentence_spans.mean_lead_in_ms << ",\n"
        << "    \"mean_text_sentence_span_tail_out_ms\": " << grade_report.pause_alignment.text_sentence_spans.mean_tail_out_ms << ",\n"
        << "    \"text_sentence_region_subspan_reference_count\": " << grade_report.pause_alignment.text_sentence_region_subspans.reference_count << ",\n"
        << "    \"text_sentence_region_subspan_predicted_count\": " << grade_report.pause_alignment.text_sentence_region_subspans.predicted_count << ",\n"
        << "    \"text_sentence_region_subspan_matched_count\": " << grade_report.pause_alignment.text_sentence_region_subspans.matched_count << ",\n"
        << "    \"text_sentence_region_subspan_missing_count\": " << grade_report.pause_alignment.text_sentence_region_subspans.missing_count << ",\n"
        << "    \"text_sentence_region_subspan_extra_count\": " << grade_report.pause_alignment.text_sentence_region_subspans.extra_count << ",\n"
        << "    \"mean_abs_text_sentence_region_subspan_start_error_ms\": " << grade_report.pause_alignment.text_sentence_region_subspans.mean_abs_start_error_ms << ",\n"
        << "    \"median_abs_text_sentence_region_subspan_start_error_ms\": " << grade_report.pause_alignment.text_sentence_region_subspans.median_abs_start_error_ms << ",\n"
        << "    \"p90_abs_text_sentence_region_subspan_start_error_ms\": " << grade_report.pause_alignment.text_sentence_region_subspans.p90_abs_start_error_ms << ",\n"
        << "    \"mean_abs_text_sentence_region_subspan_end_error_ms\": " << grade_report.pause_alignment.text_sentence_region_subspans.mean_abs_end_error_ms << ",\n"
        << "    \"median_abs_text_sentence_region_subspan_end_error_ms\": " << grade_report.pause_alignment.text_sentence_region_subspans.median_abs_end_error_ms << ",\n"
        << "    \"p90_abs_text_sentence_region_subspan_end_error_ms\": " << grade_report.pause_alignment.text_sentence_region_subspans.p90_abs_end_error_ms << ",\n"
        << "    \"text_sentence_region_subspan_count_mismatch\": " << (grade_report.pause_alignment.text_sentence_region_subspans.count_mismatch ? "true" : "false") << ",\n"
        << "    \"mean_text_sentence_region_subspan_lead_in_ms\": " << grade_report.pause_alignment.text_sentence_region_subspans.mean_lead_in_ms << ",\n"
        << "    \"mean_text_sentence_region_subspan_tail_out_ms\": " << grade_report.pause_alignment.text_sentence_region_subspans.mean_tail_out_ms << "\n"
        << "  },\n"
        << "  \"word_onset_alignment\": {\n"
        << "    \"reference_count\": " << grade_report.word_onset_alignment.reference_count << ",\n"
        << "    \"matched_count\": " << grade_report.word_onset_alignment.matched_count << ",\n"
        << "    \"missing_count\": " << grade_report.word_onset_alignment.missing_count << ",\n"
        << "    \"mean_abs_start_error_ms\": " << grade_report.word_onset_alignment.mean_abs_start_error_ms << ",\n"
        << "    \"median_abs_start_error_ms\": " << grade_report.word_onset_alignment.median_abs_start_error_ms << ",\n"
        << "    \"p90_abs_start_error_ms\": " << grade_report.word_onset_alignment.p90_abs_start_error_ms << ",\n"
        << "    \"mean_early_start_ms\": " << grade_report.word_onset_alignment.mean_early_start_ms << ",\n"
        << "    \"mean_late_start_ms\": " << grade_report.word_onset_alignment.mean_late_start_ms << "\n"
        << "  },\n"
        << "  \"intra_word_alignment\": {\n"
        << "    \"reference_count\": " << grade_report.intra_word_alignment.reference_count << ",\n"
        << "    \"matched_count\": " << grade_report.intra_word_alignment.matched_count << ",\n"
        << "    \"missing_count\": " << grade_report.intra_word_alignment.missing_count << ",\n"
        << "    \"extra_count\": " << grade_report.intra_word_alignment.extra_count << ",\n"
        << "    \"mean_abs_center_error_ms\": " << grade_report.intra_word_alignment.mean_abs_center_error_ms << ",\n"
        << "    \"median_abs_center_error_ms\": " << grade_report.intra_word_alignment.median_abs_center_error_ms << ",\n"
        << "    \"p90_abs_center_error_ms\": " << grade_report.intra_word_alignment.p90_abs_center_error_ms << ",\n"
        << "    \"mean_abs_start_error_ms\": " << grade_report.intra_word_alignment.mean_abs_start_error_ms << ",\n"
        << "    \"mean_abs_end_error_ms\": " << grade_report.intra_word_alignment.mean_abs_end_error_ms << "\n"
        << "  },\n"
        << "  \"speech_region_containment\": {\n"
        << "    \"predicted_region_count\": " << grade_report.speech_region_containment.predicted_region_count << ",\n"
        << "    \"event_count\": " << grade_report.speech_region_containment.event_count << ",\n"
        << "    \"events_with_invalid_region\": " << grade_report.speech_region_containment.events_with_invalid_region << ",\n"
        << "    \"events_outside_region\": " << grade_report.speech_region_containment.events_outside_region << ",\n"
        << "    \"events_starting_before_region\": " << grade_report.speech_region_containment.events_starting_before_region << ",\n"
        << "    \"events_ending_after_region\": " << grade_report.speech_region_containment.events_ending_after_region << ",\n"
        << "    \"mean_early_start_leak_ms\": " << grade_report.speech_region_containment.mean_early_start_leak_ms << ",\n"
        << "    \"mean_late_end_leak_ms\": " << grade_report.speech_region_containment.mean_late_end_leak_ms << ",\n"
        << "    \"max_early_start_leak_ms\": " << grade_report.speech_region_containment.max_early_start_leak_ms << ",\n"
        << "    \"max_late_end_leak_ms\": " << grade_report.speech_region_containment.max_late_end_leak_ms << "\n"
        << "  }\n"
        << "}\n";
    return out.str();
}

static std::string streaming_detector_summary_json(const StreamingDetectorReport& report)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(6);
    const double pause_accuracy = report.pause_frames > 0
        ? static_cast<double>(report.pause_matches) / static_cast<double>(report.pause_frames)
        : 0.0;
    const double boundary_precision = report.predicted_word_boundaries > 0
        ? static_cast<double>(report.matched_word_boundaries) / static_cast<double>(report.predicted_word_boundaries)
        : 0.0;
    const double boundary_recall = report.gold_word_boundaries > 0
        ? static_cast<double>(report.matched_word_boundaries) / static_cast<double>(report.gold_word_boundaries)
        : 0.0;
    const double boundary_f1 = (boundary_precision + boundary_recall) > 0.0
        ? (2.0 * boundary_precision * boundary_recall) / (boundary_precision + boundary_recall)
        : 0.0;
    out << "{\n"
        << "  \"pause_frames\": " << report.pause_frames << ",\n"
        << "  \"pause_matches\": " << report.pause_matches << ",\n"
        << "  \"pause_top1_accuracy\": " << pause_accuracy << ",\n"
        << "  \"gold_word_boundaries\": " << report.gold_word_boundaries << ",\n"
        << "  \"predicted_word_boundaries\": " << report.predicted_word_boundaries << ",\n"
        << "  \"matched_word_boundaries\": " << report.matched_word_boundaries << ",\n"
        << "  \"word_boundary_precision\": " << boundary_precision << ",\n"
        << "  \"word_boundary_recall\": " << boundary_recall << ",\n"
        << "  \"word_boundary_f1\": " << boundary_f1 << ",\n"
        << "  \"word_boundary_mean_abs_ms\": " << report.mean_abs_word_boundary_error_ms << ",\n"
        << "  \"word_boundary_median_abs_ms\": " << report.median_abs_word_boundary_error_ms << ",\n"
        << "  \"word_boundary_p90_abs_ms\": " << report.p90_abs_word_boundary_error_ms << "\n"
        << "}\n";
    return out.str();
}

static float wav_duration_seconds(const Wav& wav)
{
    if (wav.sample_rate <= 0 || wav.channels <= 0) return 0.0f;
    return static_cast<float>(wav.samples.size()) / static_cast<float>(wav.sample_rate * wav.channels);
}

struct RunnerCliConfig
{
    fs::path root = fs::current_path();
    StreamConfig stream;
    OutputConfig output;
    bool fast_batch = false;
    std::string case_filter;
    fs::path neural_track_csv;
    fs::path neural_checkpoint;
};

static RunnerCliConfig parse_cli_config(int argc, char** argv)
{
    RunnerCliConfig cli;
    bool root_set = false;
    bool tick_set = false;

    auto parse_ms = [](const std::string& value, const char* flag) {
        const double parsed = std::stod(value);
        if (parsed < 0.0)
        {
            throw std::runtime_error(std::string(flag) + " must be non-negative");
        }
        return static_cast<float>(parsed / 1000.0);
    };

    auto starts_with = [](const std::string& s, const char* prefix) {
        const std::string p(prefix);
        return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
    };

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];

        if (arg == "--root")
        {
            if (i + 1 >= argc) throw std::runtime_error("--root requires a value");
            cli.root = fs::path(argv[++i]);
            root_set = true;
        }
        else if (starts_with(arg, "--root="))
        {
            cli.root = fs::path(arg.substr(std::string("--root=").size()));
            root_set = true;
        }
        else if (arg == "--buffer-ms" || arg == "--preroll-ms" || arg == "--preroll")
        {
            if (i + 1 >= argc) throw std::runtime_error(arg + " requires a value");
            cli.stream.buffer_seconds = parse_ms(argv[++i], arg.c_str());
        }
        else if (starts_with(arg, "--buffer-ms=") || starts_with(arg, "--preroll-ms=") || starts_with(arg, "--preroll="))
        {
            const size_t eq = arg.find('=');
            const std::string flag = arg.substr(0, eq);
            cli.stream.buffer_seconds = parse_ms(arg.substr(eq + 1), flag.c_str());
        }
        else if (arg == "--evidence-postroll-ms")
        {
            if (i + 1 >= argc) throw std::runtime_error("--evidence-postroll-ms requires a value");
            cli.stream.evidence_postroll_seconds = parse_ms(argv[++i], "--evidence-postroll-ms");
        }
        else if (starts_with(arg, "--evidence-postroll-ms="))
        {
            cli.stream.evidence_postroll_seconds = parse_ms(
                arg.substr(std::string("--evidence-postroll-ms=").size()),
                "--evidence-postroll-ms");
        }
        else if (arg == "--chunk-ms")
        {
            if (i + 1 >= argc) throw std::runtime_error("--chunk-ms requires a value");
            cli.stream.chunk_seconds = parse_ms(argv[++i], "--chunk-ms");
        }
        else if (starts_with(arg, "--chunk-ms="))
        {
            cli.stream.chunk_seconds = parse_ms(arg.substr(std::string("--chunk-ms=").size()), "--chunk-ms");
        }
        else if (arg == "--tick-ms")
        {
            if (i + 1 >= argc) throw std::runtime_error("--tick-ms requires a value");
            cli.stream.playback_tick_seconds = parse_ms(argv[++i], "--tick-ms");
            tick_set = true;
        }
        else if (starts_with(arg, "--tick-ms="))
        {
            cli.stream.playback_tick_seconds = parse_ms(arg.substr(std::string("--tick-ms=").size()), "--tick-ms");
            tick_set = true;
        }
        else if (arg == "--fast-batch")
        {
            cli.fast_batch = true;
            cli.output.write_detailed_diagnostics = false;
        }
        else if (arg == "--full-diagnostics")
        {
            cli.output.write_detailed_diagnostics = true;
        }
        else if (arg == "--export-monotonic-dataset")
        {
            cli.output.write_monotonic_training_rows = true;
        }
        else if (arg == "--neural-track-csv")
        {
            if (i + 1 >= argc) throw std::runtime_error("--neural-track-csv requires a value");
            cli.neural_track_csv = fs::path(argv[++i]);
        }
        else if (starts_with(arg, "--neural-track-csv="))
        {
            cli.neural_track_csv = fs::path(arg.substr(std::string("--neural-track-csv=").size()));
        }
        else if (arg == "--neural-checkpoint")
        {
            if (i + 1 >= argc) throw std::runtime_error("--neural-checkpoint requires a value");
            cli.neural_checkpoint = fs::path(argv[++i]);
        }
        else if (starts_with(arg, "--neural-checkpoint="))
        {
            cli.neural_checkpoint = fs::path(
                arg.substr(std::string("--neural-checkpoint=").size()));
        }
        else if (arg == "--case")
        {
            if (i + 1 >= argc) throw std::runtime_error("--case requires a value");
            cli.case_filter = argv[++i];
        }
        else if (starts_with(arg, "--case="))
        {
            cli.case_filter = arg.substr(std::string("--case=").size());
        }
        else if (!starts_with(arg, "--") && !root_set)
        {
            cli.root = fs::path(arg);
            root_set = true;
        }
        else
        {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (cli.stream.chunk_seconds <= 0.0f)
    {
        throw std::runtime_error("--chunk-ms must be greater than zero");
    }
    if (cli.stream.playback_tick_seconds <= 0.0f)
    {
        throw std::runtime_error("--tick-ms must be greater than zero");
    }
    if (cli.fast_batch && !tick_set)
    {
        cli.stream.playback_tick_seconds = std::max(cli.stream.chunk_seconds, 0.040f);
    }
    return cli;
}

static int32 stream_chunk_frames(int sample_rate, float chunk_seconds)
{
    return std::max(1, static_cast<int32>(std::lround(static_cast<double>(sample_rate) * static_cast<double>(chunk_seconds))));
}

static void run_streamed_runtime_session(
    FOffgridAILipsyncRuntimeSession& session,
    const Wav& wav,
    const StreamConfig& stream)
{
    if (wav.sample_rate <= 0 || wav.channels <= 0 || wav.samples.empty())
    {
        session.CloseInputStream();
        session.Finalize(0.0f);
        return;
    }

    const float buffer_seconds = std::max(stream.buffer_seconds, 0.0f);
    const float tick_seconds = std::max(stream.playback_tick_seconds, 0.001f);
    const int32 frames_per_chunk = stream_chunk_frames(wav.sample_rate, stream.chunk_seconds);
    const int32 total_frames = static_cast<int32>(wav.samples.size() / static_cast<size_t>(wav.channels));
    int32 frame_cursor = 0;
    float playback_seconds = 0.0f;

    auto advance_playback_to = [&](float target_playback_seconds)
    {
        target_playback_seconds = std::max(target_playback_seconds, playback_seconds);
        while (playback_seconds + tick_seconds < target_playback_seconds)
        {
            playback_seconds += tick_seconds;
            session.Update(playback_seconds);
        }
        if (target_playback_seconds > playback_seconds + 0.0005f)
        {
            playback_seconds = target_playback_seconds;
            session.Update(playback_seconds);
        }
    };

    while (frame_cursor < total_frames)
    {
        const int32 chunk_frames = std::min(frames_per_chunk, total_frames - frame_cursor);
        TArray<uint8> chunk_bytes;
        const int32 int16_start = frame_cursor * wav.channels;
        const int32 int16_count = chunk_frames * wav.channels;
        const auto* raw = reinterpret_cast<const uint8*>(wav.samples.data() + int16_start);
        for (int32 i = 0; i < int16_count * static_cast<int32>(sizeof(int16)); ++i)
        {
            chunk_bytes.Add(raw[i]);
        }

        session.PushAudioPCM16(chunk_bytes, chunk_bytes.Num(), wav.sample_rate, wav.channels, frame_cursor);

        const float observed_end = static_cast<float>(frame_cursor + chunk_frames) / static_cast<float>(wav.sample_rate);
        const float target_playback_seconds = std::max(0.0f, observed_end - buffer_seconds);
        advance_playback_to(target_playback_seconds);

        frame_cursor += chunk_frames;
    }

    session.CloseInputStream();

    const float duration_seconds = wav_duration_seconds(wav);
    // Input completion is not playback completion. Drain the buffered tail
    // through ordinary Update() calls so the final preroll window can resolve
    // boundaries and commit live events before Finalize() records terminal
    // diagnostics. Skipping this interval manufactured late suffix bursts that
    // the real renderer could never present.
    advance_playback_to(duration_seconds);

    session.Finalize(duration_seconds);
}

int main(int argc, char** argv)
{
    try
    {
        using clock = std::chrono::steady_clock;
        const RunnerCliConfig cli = parse_cli_config(argc, argv);
        const PredictedNeuralTrackByCase predicted_neural_tracks = cli.neural_track_csv.empty()
            ? PredictedNeuralTrackByCase{}
            : read_predicted_neural_track_csv(fs::absolute(cli.neural_track_csv));
        const fs::path root = cli.root;
        const StreamConfig stream = cli.stream;
        const OutputConfig output = cli.output;
        fs::path out_root = root / "outputs" / "runs" / "latest";
        fs::remove_all(out_root);
        fs::create_directories(out_root);

        std::vector<fs::path> transcript_paths;
        for (auto& entry : fs::directory_iterator(root / "inputs" / "transcripts"))
        {
            if (entry.is_regular_file() && entry.path().extension() == ".txt")
            {
                transcript_paths.push_back(entry.path());
            }
        }
        std::sort(transcript_paths.begin(), transcript_paths.end());
        if (!cli.case_filter.empty())
        {
            transcript_paths.erase(
                std::remove_if(
                    transcript_paths.begin(),
                    transcript_paths.end(),
                    [&](const fs::path& path) {
                        return path.stem().string().find(cli.case_filter) == std::string::npos;
                    }),
                transcript_paths.end());
        }

        const auto run_started = clock::now();
        const int total_cases = static_cast<int>(transcript_paths.size());
        write_progress_json(out_root, "running_corpus", 0, total_cases, "", 0.0);

        int cases = 0;
        bool all_ok = true;
        for (const fs::path& transcript_path : transcript_paths)
        {
            const std::string stem = transcript_path.stem().string();
            const fs::path wav_path = root / "inputs" / "wav" / (stem + ".wav");
            if (!fs::exists(wav_path))
            {
                std::cerr << "missing wav for " << stem << "\n";
                all_ok = false;
                continue;
            }

            const auto case_started = clock::now();
            const std::string transcript = read_text(transcript_path);
            const Wav wav = read_wav(wav_path);

            FOffgridAILipsyncRuntimeSession session;
            FOffgridAILipsyncRuntimeBeginInput begin;
            begin.DialogueText = FString(transcript);
            begin.LineID = FName(stem);
            begin.NPCID = FName("liplab");
            begin.PrerollSec = stream.buffer_seconds;
            if (!cli.neural_checkpoint.empty())
                begin.NeuralCheckpointPath = FString(cli.neural_checkpoint.string());
            if (!cli.case_filter.empty()) std::cerr << "[case] begin " << stem << std::endl;
            session.BeginLine(begin);
            if (!cli.case_filter.empty()) std::cerr << "[case] stream " << stem << std::endl;

            run_streamed_runtime_session(session, wav, stream);
            if (!cli.case_filter.empty()) std::cerr << "[case] export " << stem << std::endl;

            const auto& plan = session.GetTextPlan();
            const auto& speech = session.GetSpeechRegions();
            const auto& gap_candidates = session.GetSpeechDetector().GetRefinedGapCandidates();
            const auto& committed = session.GetCommittedTrack();

            const fs::path case_dir = out_root / stem;
            fs::remove_all(case_dir);
            fs::create_directories(case_dir);
            write_text(case_dir / "planned.csv", planned_csv(plan));
            write_text(case_dir / "speech_regions.csv", speech_csv(speech));
            write_text(case_dir / "committed.csv", committed_csv(committed));
            write_text(case_dir / "dropped.csv", dropped_csv(committed));
            write_text(case_dir / "region_drop_diagnostics.csv", region_drop_diagnostics_csv(committed, speech));
            write_text(case_dir / "presentation_motion_grade.json",
                presentation_motion_grade_json(committed));
            if (output.write_detailed_diagnostics)
            {
                write_text(case_dir / "gap_candidates.csv", gap_candidates_csv(gap_candidates));
                write_text(case_dir / "soft_lull_candidates.csv", soft_lull_candidates_csv(session.GetSpeechDetector().GetSoftLullCandidates()));
                write_text(case_dir / "occupancy_frames.csv", occupancy_frames_csv(session.GetSpeechDetector().GetFeatureFrames()));
                write_text(case_dir / "phone_class_frames.csv", phone_class_frames_csv(session.GetSpeechDetector().GetFeatureFrames()));
                write_text(case_dir / "commit_decisions.csv", commit_decisions_csv(committed));
                write_text(case_dir / "runtime_speech_regions.csv", runtime_speech_regions_csv(session.GetRuntimeSpeechRegionDiagnosticRows()));
                write_text(case_dir / "runtime_boundary_state.csv", runtime_boundary_state_csv(session.GetRuntimeBoundaryDiagnosticRows()));
                write_text(case_dir / "stream_tail.csv", stream_tail_csv(session.GetStreamTailDiagnosticRow()));
            }
            GradeReport report;
            // Raw runtime evidence is useful when diagnosing newly captured
            // Offgrid lines before they have been promoted into checked-in
            // gold. Keep this host-side export independent of gold grading.
            if (output.write_detailed_diagnostics)
            {
                FOffgridAIStreamingEvidenceSurfaceConfig evidence_surface_config;
                evidence_surface_config.PrerollSec = stream.buffer_seconds;
                evidence_surface_config.PostrollSec = stream.evidence_postroll_seconds;
                evidence_surface_config.SpeechRegions = &speech;
                const auto evidence_observations =
                    FOffgridAIStreamingEvidenceSurface::Analyze(
                        session.GetSpeechDetector().GetFeatureFrames(),
                        evidence_surface_config);
                write_text(case_dir / "streaming_evidence_observations.csv",
                    evidence_ingredient_observations_csv(evidence_observations));
                const auto syllable_candidate_sets =
                    FOffgridAIStreamingSyllablePositionEstimator::EstimateCandidateSets(
                        plan, evidence_observations);
                write_text(case_dir / "streaming_syllable_candidate_sets.csv",
                    syllable_candidate_sets_csv(syllable_candidate_sets));
            }
            const fs::path gold_case_dir = root / "inputs" / "gold" / stem;
            const fs::path gold_phones_path = gold_case_dir / "phones.csv";
            const fs::path gold_words_path = gold_case_dir / "words.csv";
            const fs::path gold_speech_path = gold_case_dir / "speech.csv";
            const fs::path gold_boundaries_path = gold_case_dir / "boundaries.csv";
            const bool gold_available =
                fs::exists(gold_phones_path)
                && fs::exists(gold_words_path)
                && fs::exists(gold_speech_path)
                && fs::exists(gold_boundaries_path);
            size_t gold_viseme_count = 0;
            if (gold_available)
            {
                if (!cli.case_filter.empty()) std::cerr << "[case] read-gold " << stem << std::endl;
                const auto gold_phones = read_gold_phones_csv(gold_phones_path);
                const auto gold_words = read_gold_words_csv(gold_words_path);
                const auto gold_speech = read_gold_speech_csv(gold_speech_path);
                const auto gold_boundaries = read_gold_boundaries_csv(gold_boundaries_path);
                FOffgridAIStreamingEvidenceSurfaceConfig evidence_surface_config;
                evidence_surface_config.PrerollSec = stream.buffer_seconds;
                evidence_surface_config.PostrollSec = stream.evidence_postroll_seconds;
                evidence_surface_config.SpeechRegions = &speech;
                const TArray<FOffgridAIAudioLandmarkObservation> evidence_observations =
                    FOffgridAIStreamingEvidenceSurface::Analyze(
                        session.GetSpeechDetector().GetFeatureFrames(),
                        evidence_surface_config);
                const TArray<FOffgridAIAudioLandmarkObservation> permissive_evidence_candidates =
                    evidence_observations;
                const std::vector<EvidenceIngredientTarget> evidence_targets =
                    build_evidence_ingredient_targets(gold_phones, gold_words, gold_speech, gold_boundaries);
                const EvidenceIngredientReport evidence_report = grade_evidence_ingredients(
                    evidence_targets,
                    evidence_observations,
                    gold_speech,
                    evidence_surface_config.PrerollSec * 1000.0,
                    evidence_surface_config.PostrollSec * 1000.0);
                const EvidenceIngredientReport intra_envelope_phone_report =
                    grade_intra_envelope_phone_evidence(
                        plan,
                        gold_phones,
                        permissive_evidence_candidates,
                        evidence_surface_config.PrerollSec * 1000.0,
                        evidence_surface_config.PostrollSec * 1000.0);
                write_text(case_dir / "streaming_evidence_observations.csv",
                    evidence_ingredient_observations_csv(evidence_observations));
                write_text(case_dir / "streaming_evidence_grade.json",
                    evidence_ingredient_grade_json(evidence_report));
                write_text(case_dir / "streaming_evidence_candidates.csv",
                    evidence_ingredient_observations_csv(permissive_evidence_candidates));
                write_text(case_dir / "streaming_intra_envelope_phone_grade.json",
                    evidence_ingredient_grade_json(intra_envelope_phone_report));
                if (!cli.case_filter.empty()) std::cerr << "[case] landmark-targets " << stem << std::endl;
                if (!cli.case_filter.empty()) std::cerr << "[case] prosody-targets " << stem << std::endl;
                const auto prosodic_peak_targets = build_prosodic_peak_targets(plan, gold_phones);
                const auto mfa_nucleus_targets = build_mfa_nucleus_targets(gold_phones);
                const TArray<FOffgridAIStreamingSyllableCandidateSet> syllable_candidate_sets =
                    FOffgridAIStreamingSyllablePositionEstimator::EstimateCandidateSets(
                        plan,
                        permissive_evidence_candidates);
                const SyllableCandidateSetReport syllable_candidate_set_report =
                    grade_syllable_candidate_sets(
                        prosodic_peak_targets,
                        syllable_candidate_sets);
                write_text(case_dir / "streaming_syllable_candidate_sets.csv",
                    syllable_candidate_sets_csv(syllable_candidate_sets));
                write_text(case_dir / "streaming_syllable_candidate_set_grade.json",
                    syllable_candidate_set_grade_json(syllable_candidate_set_report));
                if (!cli.case_filter.empty()) std::cerr << "[case] visible-labels " << stem << std::endl;
                const auto handmade = build_gold_visible_labels(plan, gold_phones);
                gold_viseme_count = handmade.size();
                if (output.write_monotonic_training_rows)
                {
                    write_text(
                        case_dir / "monotonic_audio_frames.csv",
                        monotonic_audio_frames_csv(session.GetSpeechDetector().GetFeatureFrames()));
                    write_text(
                        case_dir / "monotonic_tokens.csv",
                        monotonic_tokens_csv(plan, handmade, gold_words));
                }
                if (!cli.case_filter.empty()) std::cerr << "[case] grade " << stem << std::endl;
                write_text(case_dir / "planned_words.csv", planned_words_csv(plan));
                write_text(case_dir / "expected_phones.csv", expected_phones_csv(plan));
                write_text(case_dir / "planned_syllables.csv", planned_syllables_csv(plan));
                write_text(case_dir / "planned_boundaries.csv", planned_boundaries_csv(plan));
                write_text(case_dir / "gold_visible_visemes.csv", gold_visible_visemes_csv(handmade));
                write_text(case_dir / "focus_alignment_grade.json",
                    focus_alignment_grade_json(committed, plan, handmade, gold_speech));
                write_text(case_dir / "runtime_delivery_grade.json",
                    runtime_delivery_grade_json(
                        committed, plan, gold_words, gold_speech));
                write_text(case_dir / "region_ownership_grade.json",
                    region_ownership_grade_json(committed, plan, gold_words, gold_speech));
                report = grade(committed, speech, handmade, gold_words, gold_speech);
                const auto neural_prediction = predicted_neural_tracks.find(stem);
                if (neural_prediction != predicted_neural_tracks.end())
                {
                    const PredictedNeuralTrackResult neural = build_predicted_neural_track(
                        plan,
                        neural_prediction->second);
                    const GradeReport neural_grade = grade(
                        neural.track,
                        neural.speech_regions,
                        handmade,
                        gold_words,
                        gold_speech);
                    write_text(
                        case_dir / "neural_track_summary.json",
                        predicted_neural_track_summary_json(neural, neural_grade));
                    write_text(case_dir / "neural_track_grade.json", grade_json(neural_grade));
                    auto neural_syllables = neural_syllable_observations(
                        prosodic_peak_targets, neural.track);
                    const ProsodicPeakReport neural_syllable_report =
                        grade_monotonic_neural_syllables(mfa_nucleus_targets, neural_syllables);
                    write_text(case_dir / "neural_syllable_alignment_grade.json",
                        prosodic_peak_grade_json(neural_syllable_report));
                    write_text(case_dir / "neural_error_samples.json",
                        neural_error_samples_json(neural_grade, neural_syllable_report));
                }
                if (output.write_detailed_diagnostics)
                {
                    write_text(case_dir / "audio_landmark_frames.csv", audio_landmark_frames_csv(session.GetSpeechDetector().GetFeatureFrames()));
                }
                std::vector<ProsodicPeakObservation> runtime_syllable_observations =
                    runtime_syllable_assignment_observations(
                        prosodic_peak_targets,
                        session.GetRuntimeSyllableAssignmentDiagnosticRows());
                const ProsodicPeakReport runtime_syllable_report = grade_prosodic_peaks(
                    prosodic_peak_targets,
                    runtime_syllable_observations,
                    std::vector<ProsodicPeakObservation>());
                const auto word_start_targets =
                    word_start_nucleus_targets(prosodic_peak_targets);
                auto runtime_word_start_observations =
                    runtime_word_start_nucleus_observations(
                        word_start_targets,
                        session.GetRuntimeSyllableAssignmentDiagnosticRows(),
                        committed);
                const ProsodicPeakReport runtime_word_start_report =
                    grade_prosodic_peaks(
                        word_start_targets,
                        runtime_word_start_observations,
                        std::vector<ProsodicPeakObservation>());
                const auto performed_word_head_observations =
                    performed_word_head_anchor_observations(committed);
                std::vector<ProsodicPeakObservation> no_assigned_word_heads;
                const ProsodicPeakReport nearby_nucleus_report =
                    grade_prosodic_peaks(
                        mfa_nucleus_targets,
                        no_assigned_word_heads,
                        performed_word_head_observations);
                write_text(case_dir / "runtime_syllable_anchor_diagnostics.csv",
                    runtime_syllable_anchor_diagnostics_csv(
                        session.GetRuntimeSyllableAssignmentDiagnosticRows()));
                write_text(case_dir / "runtime_syllable_assignment_grade.json",
                    prosodic_peak_grade_json(runtime_syllable_report));
                write_text(case_dir / "runtime_word_start_nucleus_grade.json",
                    prosodic_peak_grade_json(runtime_word_start_report));
                write_text(case_dir / "performed_word_head_nearby_nucleus_grade.json",
                    prosodic_peak_grade_json(nearby_nucleus_report));
                write_text(case_dir / "word_onset_diagnostics.csv", word_onset_diagnostics_csv(committed, handmade, gold_words));
                if (!cli.case_filter.empty()) std::cerr << "[case] reports-done " << stem << std::endl;
            }
            else
            {
                report.gold_available = false;
                report.gold_status = "missing";
                report.committed_count = committed.Events.Num();
            }
            write_text(case_dir / "grade.json", grade_json(report));

            const double case_seconds = std::chrono::duration<double>(clock::now() - case_started).count();
            const double elapsed_seconds = std::chrono::duration<double>(clock::now() - run_started).count();
            write_progress_json(out_root, "running_corpus", cases + 1, total_cases, stem, elapsed_seconds, case_seconds);

            std::cout << "[" << (cases + 1) << "/" << total_cases << "] " << stem
                      << ": planned=" << plan.Events.Num()
                      << " committed=" << committed.Events.Num()
                      << " gold=" << gold_viseme_count
                      << " mean_ms=" << report.mean_abs_center_error_ms
                      << " case_s=" << std::fixed << std::setprecision(2) << case_seconds
                      << " elapsed_s=" << elapsed_seconds << "\n";
            ++cases;
        }

        if (cases == 0)
        {
            std::cerr << "no transcript cases found\n";
            write_progress_json(out_root, "failed_no_cases", 0, total_cases, "", 0.0);
            return 2;
        }
        write_progress_json(
            out_root,
            all_ok ? "completed" : "completed_with_errors",
            cases,
            total_cases,
            "",
            std::chrono::duration<double>(clock::now() - run_started).count());
        return all_ok ? 0 : 1;
    }
    catch (const std::exception& error)
    {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
