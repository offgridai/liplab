#include "Lipsync/OffgridAILipsyncRuntimeAdapter.h"
#include "Lipsync/OffgridAIOnlinePhoneAligner.h"
#include "Lipsync/OffgridAIVisemePerformer.h"

#include <cstring>
#include <cctype>
#include <algorithm>
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

struct LandmarkTarget
{
    int target_index = -1;
    std::string type;
    int expected_phone_index = -1;
    int global_phone_index = -1;
    int word_index = -1;
    int next_word_index = -1;
    int speech_region_index = -1;
    int sentence_index = -1;
    std::string word;
    std::string next_word;
    std::string phone;
    std::string phone_base;
    std::string boundary_mark;
    std::string pause_class;
    double gold_start = 0.0;
    double gold_end = 0.0;
    double gold_center = 0.0;
    double prior_start = 0.0;
    double prior_end = 0.0;
    double prior_center = 0.0;
    double prior_duration = 0.0;
    bool has_gold_reference = false;
};

struct LandmarkObservation
{
    int target_index = -1;
    std::string type;
    int frame_index = -1;
    double start = 0.0;
    double end = 0.0;
    double center = 0.0;
    double score = 0.0;
    double confidence = 0.0;
    double decision_sec = 0.0;
    std::string top_class;
    std::string source;
    int matched_target_index = -1;
    double matched_error_ms = 0.0;
};

struct LandmarkTypeReport
{
    int target_count = 0;
    int observation_count = 0;
    int matched_target_count = 0;
    int matched_observation_count = 0;
    double recall = 0.0;
    double precision = 0.0;
    double mean_abs_center_error_ms = 0.0;
    double median_abs_center_error_ms = 0.0;
    double p90_abs_center_error_ms = 0.0;
    double mean_target_window_peak_score = 0.0;
    double mean_matched_observation_score = 0.0;
    double target_window_hit_rate = 0.0;
    std::map<std::string, int> threshold_observation_count;
    std::map<std::string, int> threshold_matched_observation_count;
    std::map<std::string, int> threshold_matched_target_count;
};

struct BoundaryEventAuditReport
{
    int target_count = 0;
    int observation_count = 0;
    int matched_target_count = 0;
    int matched_observation_count = 0;
    double precision = 0.0;
    double recall = 0.0;
    double mean_abs_error_ms = 0.0;
    double median_abs_error_ms = 0.0;
    double p90_abs_error_ms = 0.0;
    double mean_decision_latency_ms = 0.0;
    double median_decision_latency_ms = 0.0;
    double p90_decision_latency_ms = 0.0;
};

struct MonotonicBoundaryMatch
{
    int observation_index = -1;
    int target_index = -1;
    double error_ms = 0.0;
    double latency_ms = 0.0;
};

struct LandmarkAuditReport
{
    bool available = false;
    int target_count = 0;
    int observation_count = 0;
    int matched_target_count = 0;
    int matched_observation_count = 0;
    std::map<std::string, LandmarkTypeReport> by_type;
    std::map<std::string, BoundaryEventAuditReport> boundary_events;
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
    int raw_matched_count = 0;
    double raw_precision = 0.0;
    double raw_recall = 0.0;
    double raw_mean_abs_error_ms = 0.0;
};

struct BoundaryAlignmentReport
{
    int reference_count = 0;
    int predicted_count = 0;
    int matched_count = 0;
    int missing_count = 0;
    int extra_count = 0;
    double mean_abs_start_error_ms = 0.0;
    double median_abs_start_error_ms = 0.0;
    double p90_abs_start_error_ms = 0.0;
    double mean_abs_end_error_ms = 0.0;
    double median_abs_end_error_ms = 0.0;
    double p90_abs_end_error_ms = 0.0;
    double mean_lead_in_ms = 0.0;
    double mean_tail_out_ms = 0.0;
    bool count_mismatch = false;
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
};

struct GradeMatch
{
    int label_index = -1;
    int event_index = -1;
};

struct StreamConfig
{
    float buffer_seconds = 0.350f;
    float chunk_seconds = 0.040f;
    float playback_tick_seconds = 0.020f;
};

struct OutputConfig
{
    bool write_detailed_diagnostics = true;
};

struct OraclePhoneDurationReport
{
    int matched_count = 0;
    int exact_global_matches = 0;
    int fallback_word_matches = 0;
    int unmatched_count = 0;
    double mean_duration_seconds = 0.0;
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

static std::string read_text(const fs::path& path)
{
    std::ifstream file(path);
    std::ostringstream out;
    out << file.rdbuf();
    return out.str();
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

static std::optional<std::string> transcript_landmark_type_for_phone_base(const std::string& phone_base)
{
    if (phone_base == "M" || phone_base == "B" || phone_base == "P")
    {
        return std::string("mbp");
    }
    if (phone_base == "F" || phone_base == "V")
    {
        return std::string("fv");
    }
    if (phone_base == "W")
    {
        return std::string("w");
    }
    if (phone_base == "CH" || phone_base == "JH" || phone_base == "SH" || phone_base == "ZH")
    {
        return std::string("chjjsh");
    }
    if (phone_base == "UW" || phone_base == "UH" || phone_base == "OW" || phone_base == "OY" || phone_base == "AO")
    {
        return std::string("round");
    }
    return std::nullopt;
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
    int speech_region_index_col = has_header ? header_index(header, "speech_region_index") : 8;
    if (speech_region_index_col < 0 && has_header)
    {
        speech_region_index_col = header_index(header, "phrase_index");
    }
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
    int speech_region_index_col = has_header ? header_index(header, "speech_region_index") : 4;
    int sentence_index_col = has_header ? header_index(header, "text_sentence_index") : 5;
    if (speech_region_index_col < 0 && has_header)
    {
        speech_region_index_col = header_index(header, "phrase_index");
    }
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
    return landmark_score_for_type(type, FOffgridAIOnlinePhoneAligner::BuildArticulatoryProbabilityField(frame));
}

static double conditioned_landmark_template_score(
    const std::string& type,
    const TArray<FOffgridAIStreamingAudioFeatureFrame>& frames,
    int32 index)
{
    if (!frames.IsValidIndex(index))
    {
        return 0.0;
    }

    auto frame_score = [&](int32 sample_index) -> double {
        return frames.IsValidIndex(sample_index)
            ? landmark_score_for_frame(type, frames[sample_index])
            : 0.0;
    };
    auto speech_evidence = [&](int32 sample_index) -> double {
        return frames.IsValidIndex(sample_index)
            ? static_cast<double>(frames[sample_index].SpeechEvidence)
            : 0.0;
    };
    auto rms_norm = [&](int32 sample_index) -> double {
        return frames.IsValidIndex(sample_index)
            ? static_cast<double>(frames[sample_index].RMSNorm)
            : 0.0;
    };
    auto flux = [&](int32 sample_index) -> double {
        return frames.IsValidIndex(sample_index)
            ? static_cast<double>(frames[sample_index].Flux)
            : 0.0;
    };
    auto periodicity = [&](int32 sample_index) -> double {
        return frames.IsValidIndex(sample_index)
            ? static_cast<double>(frames[sample_index].Periodicity)
            : 0.0;
    };
    auto high_band = [&](int32 sample_index) -> double {
        return frames.IsValidIndex(sample_index)
            ? static_cast<double>(frames[sample_index].HighBandNorm)
            : 0.0;
    };
    auto local_max = [&](int32 lo, int32 hi) -> double {
        double best = 0.0;
        for (int32 sample_index = lo; sample_index <= hi; ++sample_index)
        {
            best = std::max(best, frame_score(sample_index));
        }
        return best;
    };
    auto local_mean = [&](int32 lo, int32 hi) -> double {
        double sum = 0.0;
        int32 count = 0;
        for (int32 sample_index = lo; sample_index <= hi; ++sample_index)
        {
            if (!frames.IsValidIndex(sample_index))
            {
                continue;
            }
            sum += frame_score(sample_index);
            ++count;
        }
        return count > 0 ? (sum / static_cast<double>(count)) : 0.0;
    };

    const auto& center_frame = frames[index];
    const FOffgridAIArticulatoryProbabilityField center_field =
        FOffgridAIOnlinePhoneAligner::BuildArticulatoryProbabilityField(center_frame);
    const double center_score = frame_score(index);

    if (type == "mbp")
    {
        const double closure_before = local_max(index - 2, index);
        const double release_after = std::max(
            static_cast<double>(center_field.Release),
            std::max(frame_score(index + 1), frame_score(index + 2)));
        const double speech_rise = clamp01((speech_evidence(index + 1) - speech_evidence(index - 1)) * 1.6);
        const double burst = std::max(
            static_cast<double>(center_field.PhoneScores.StopBurst),
            clamp01(flux(index + 1) * 1.5));
        return clamp01(
            center_score * 0.38 +
            closure_before * 0.20 +
            release_after * 0.20 +
            speech_rise * 0.12 +
            burst * 0.10);
    }

    if (type == "fv")
    {
        const double sustained = local_mean(index - 1, index + 1);
        const double friction = std::max(
            static_cast<double>(center_field.Fricative),
            local_max(index - 1, index + 1));
        const double high_noise = std::max(high_band(index), high_band(index + 1));
        return clamp01(
            center_score * 0.42 +
            sustained * 0.26 +
            friction * 0.20 +
            high_noise * 0.12);
    }

    if (type == "w")
    {
        const double sustained = local_mean(index - 1, index + 1);
        const double sonorant = std::max(
            static_cast<double>(center_field.Sonorant),
            static_cast<double>(center_field.PhoneScores.Glide));
        const double smooth = clamp01((0.12 - flux(index)) / 0.12);
        const double voicing = std::max(periodicity(index), periodicity(index + 1));
        return clamp01(
            center_score * 0.38 +
            sustained * 0.22 +
            sonorant * 0.18 +
            smooth * 0.10 +
            voicing * 0.12);
    }

    if (type == "chjjsh")
    {
        const double sustained = local_mean(index - 1, index + 1);
        const double burst = std::max(
            static_cast<double>(center_field.PhoneScores.StopBurst),
            clamp01(flux(index) * 1.4));
        return clamp01(
            center_score * 0.46 +
            sustained * 0.24 +
            burst * 0.18 +
            high_band(index) * 0.12);
    }

    if (type == "round")
    {
        const double sustained = local_mean(index - 2, index + 2);
        const double voicing = std::max(periodicity(index), periodicity(index - 1));
        const double smooth = clamp01((0.10 - flux(index)) / 0.10);
        return clamp01(
            center_score * 0.34 +
            sustained * 0.34 +
            voicing * 0.16 +
            smooth * 0.16);
    }

    if (type == "comma_lull")
    {
        const double lull_center = center_score;
        const double lull_sustain = local_mean(index - 1, index + 1);
        const double reopen_speech = std::max({
            speech_evidence(index + 1),
            speech_evidence(index + 2),
            speech_evidence(index + 3)
        });
        const double quiet_floor = clamp01((0.08 - rms_norm(index)) / 0.08);
        return clamp01(
            lull_center * 0.40 +
            lull_sustain * 0.24 +
            reopen_speech * 0.18 +
            quiet_floor * 0.18);
    }

    return center_score;
}

static double conditioned_landmark_target_score(
    const LandmarkTarget& target,
    const TArray<FOffgridAIStreamingAudioFeatureFrame>& frames,
    int32 index)
{
    if (!frames.IsValidIndex(index))
    {
        return 0.0;
    }

    const double family_score = conditioned_landmark_template_score(target.type, frames, index);
    if (family_score <= 0.0)
    {
        return 0.0;
    }

    const auto& frame = frames[index];
    const FOffgridAIArticulatoryProbabilityField field =
        FOffgridAIOnlinePhoneAligner::BuildArticulatoryProbabilityField(frame);
    const std::string phone_base = target.phone_base;
    auto local_mean = [&](int32 lo, int32 hi, const auto& getter) -> double {
        double sum = 0.0;
        int32 count = 0;
        for (int32 sample_index = lo; sample_index <= hi; ++sample_index)
        {
            if (!frames.IsValidIndex(sample_index))
            {
                continue;
            }
            sum += getter(frames[sample_index]);
            ++count;
        }
        return count > 0 ? (sum / static_cast<double>(count)) : 0.0;
    };
    auto local_max = [&](int32 lo, int32 hi, const auto& getter) -> double {
        double best = 0.0;
        for (int32 sample_index = lo; sample_index <= hi; ++sample_index)
        {
            if (!frames.IsValidIndex(sample_index))
            {
                continue;
            }
            best = std::max(best, getter(frames[sample_index]));
        }
        return best;
    };

    const double voiced = static_cast<double>(field.Voiced);
    const double closure = static_cast<double>(field.Closure);
    const double release = static_cast<double>(field.Release);
    const double fricative = static_cast<double>(field.Fricative);
    const double sonorant = static_cast<double>(field.Sonorant);
    const double bilabial = static_cast<double>(field.PhoneScores.Bilabial);
    const double labiodental = static_cast<double>(field.PhoneScores.Labiodental);
    const double sibilant = static_cast<double>(field.PhoneScores.Sibilant);
    const double burst = static_cast<double>(field.PhoneScores.StopBurst);
    const double glide = static_cast<double>(field.PhoneScores.Glide);
    const double liquid = static_cast<double>(field.PhoneScores.Liquid);
    const double nasal = static_cast<double>(field.PhoneScores.Nasal);
    const double vowel_round = static_cast<double>(field.PhoneScores.VowelRound);
    const double high_band = static_cast<double>(frame.HighBandNorm);
    const double low_band = static_cast<double>(frame.LowBandNorm);
    const double periodicity = static_cast<double>(frame.Periodicity);
    const double flux = static_cast<double>(frame.Flux);
    const double rms = static_cast<double>(frame.RMSNorm);
    const double speech = static_cast<double>(frame.SpeechEvidence);

    const double prev_rms = local_mean(index - 3, index - 1, [](const auto& f) { return static_cast<double>(f.RMSNorm); });
    const double next_rms = local_mean(index + 1, index + 3, [](const auto& f) { return static_cast<double>(f.RMSNorm); });
    const double prev_speech = local_mean(index - 3, index - 1, [](const auto& f) { return static_cast<double>(f.SpeechEvidence); });
    const double next_speech = local_mean(index + 1, index + 3, [](const auto& f) { return static_cast<double>(f.SpeechEvidence); });
    const double local_flux_peak = local_max(index - 1, index + 1, [](const auto& f) { return static_cast<double>(f.Flux); });
    const double local_periodicity = local_mean(index - 1, index + 1, [](const auto& f) { return static_cast<double>(f.Periodicity); });
    const double local_high_band = local_mean(index - 1, index + 1, [](const auto& f) { return static_cast<double>(f.HighBandNorm); });
    const double local_rise = clamp01((next_rms - prev_rms) * 2.2);
    const double local_fall = clamp01((prev_rms - next_rms) * 2.2);
    const double local_peak_shape = clamp01((rms - 0.5 * (prev_rms + next_rms)) / 0.18);
    const double local_valley_shape = clamp01((0.5 * (prev_rms + next_rms) - rms) / 0.18);
    const double voiced_stability = clamp01(local_periodicity * 0.70 + speech * 0.30 - local_flux_peak * 0.28);
    const double noisy_stability = clamp01((1.0 - local_periodicity) * 0.45 + local_high_band * 0.30 + local_flux_peak * 0.25);
    const double burst_support = clamp01(local_flux_peak * 0.58 + local_rise * 0.24 + high_band * 0.18);

    double subtype_score = family_score;
    if (phone_base == "M")
    {
        const double anti_burst = clamp01(1.0 - burst);
        const double anti_release = clamp01(1.0 - release);
        subtype_score = clamp01(
            family_score * 0.22 +
            nasal * 0.28 +
            voiced * 0.14 +
            sonorant * 0.12 +
            closure * 0.10 +
            anti_burst * 0.05 +
            anti_release * 0.05 +
            voiced_stability * 0.04);
    }
    else if (phone_base == "B")
    {
        subtype_score = clamp01(
            family_score * 0.30 +
            bilabial * 0.18 +
            closure * 0.16 +
            release * 0.12 +
            burst * 0.08 +
            voiced * 0.10 +
            burst_support * 0.06);
    }
    else if (phone_base == "P")
    {
        const double unvoiced = clamp01(1.0 - voiced);
        subtype_score = clamp01(
            family_score * 0.28 +
            bilabial * 0.18 +
            closure * 0.14 +
            release * 0.14 +
            burst * 0.10 +
            unvoiced * 0.10 +
            burst_support * 0.06);
    }
    else if (phone_base == "F")
    {
        const double unvoiced = clamp01(1.0 - voiced);
        subtype_score = clamp01(
            family_score * 0.28 +
            labiodental * 0.18 +
            fricative * 0.16 +
            high_band * 0.08 +
            unvoiced * 0.14 +
            clamp01(1.0 - low_band) * 0.10 +
            noisy_stability * 0.06);
    }
    else if (phone_base == "V")
    {
        subtype_score = clamp01(
            family_score * 0.28 +
            labiodental * 0.18 +
            fricative * 0.16 +
            high_band * 0.06 +
            voiced * 0.16 +
            periodicity * 0.10 +
            noisy_stability * 0.06);
    }
    else if (phone_base == "W")
    {
        subtype_score = clamp01(
            family_score * 0.52 +
            glide * 0.14 +
            vowel_round * 0.08 +
            sonorant * 0.06 +
            voiced * 0.05 +
            clamp01(1.0 - flux) * 0.05 +
            voiced_stability * 0.06 +
            local_peak_shape * 0.04);
    }
    else if (phone_base == "CH")
    {
        const double unvoiced = clamp01(1.0 - voiced);
        subtype_score = clamp01(
            family_score * 0.26 +
            sibilant * 0.18 +
            burst * 0.14 +
            release * 0.10 +
            high_band * 0.08 +
            unvoiced * 0.18 +
            burst_support * 0.06);
    }
    else if (phone_base == "JH")
    {
        subtype_score = clamp01(
            family_score * 0.26 +
            sibilant * 0.18 +
            burst * 0.12 +
            release * 0.08 +
            voiced * 0.18 +
            periodicity * 0.12 +
            burst_support * 0.06);
    }
    else if (phone_base == "SH")
    {
        const double unvoiced = clamp01(1.0 - voiced);
        subtype_score = clamp01(
            family_score * 0.26 +
            sibilant * 0.22 +
            fricative * 0.12 +
            high_band * 0.12 +
            unvoiced * 0.14 +
            clamp01(1.0 - burst) * 0.08 +
            noisy_stability * 0.06);
    }
    else if (phone_base == "ZH")
    {
        subtype_score = clamp01(
            family_score * 0.26 +
            sibilant * 0.20 +
            fricative * 0.12 +
            high_band * 0.10 +
            voiced * 0.16 +
            periodicity * 0.10 +
            noisy_stability * 0.06);
    }
    else if (phone_base == "UW" || phone_base == "UH")
    {
        subtype_score = clamp01(
            family_score * 0.58 +
            vowel_round * 0.12 +
            voiced * 0.08 +
            clamp01(1.0 - flux) * 0.08 +
            voiced_stability * 0.08 +
            local_peak_shape * 0.06);
    }
    else if (phone_base == "OW" || phone_base == "OY" || phone_base == "AO")
    {
        subtype_score = clamp01(
            family_score * 0.54 +
            vowel_round * 0.10 +
            voiced * 0.08 +
            static_cast<double>(field.Transition) * 0.08 +
            clamp01(1.0 - flux) * 0.06 +
            voiced_stability * 0.08 +
            local_peak_shape * 0.06);
    }
    else if (is_pause_lull_type(target.type))
    {
        const double reopen = std::max(next_speech, static_cast<double>(frames.IsValidIndex(index + 4) ? frames[index + 4].SpeechEvidence : 0.0f));
        const double pre_speech = std::max(prev_speech, static_cast<double>(frames.IsValidIndex(index - 4) ? frames[index - 4].SpeechEvidence : 0.0f));
        subtype_score = clamp01(
            family_score * 0.40 +
            reopen * 0.16 +
            pre_speech * 0.16 +
            clamp01((0.10 - rms) / 0.10) * 0.12 +
            local_valley_shape * 0.10 +
            local_fall * 0.03 +
            local_rise * 0.03);
    }

    double family_competitor = 0.0;
    for (const std::string& candidate_type : {"mbp", "fv", "w", "chjjsh", "round", "pause_lull"})
    {
        if (candidate_type == target.type)
        {
            continue;
        }
        family_competitor = std::max(
            family_competitor,
            conditioned_landmark_template_score(candidate_type, frames, index));
    }
    const double family_margin = clamp01((family_score - family_competitor + 0.12) / 0.32);
    return clamp01(subtype_score * 0.82 + family_margin * 0.18);
}

static double conditioned_landmark_competitor_score(
    const LandmarkTarget& target,
    const TArray<FOffgridAIStreamingAudioFeatureFrame>& frames,
    int32 index)
{
    double best = 0.0;
    for (const std::string& candidate_type : {"mbp", "fv", "w", "chjjsh", "round", "pause_lull"})
    {
        if (candidate_type == target.type)
        {
            continue;
        }
        best = std::max(best, conditioned_landmark_template_score(candidate_type, frames, index));
    }
    return best;
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

static void build_expected_phone_prior_times(
    const FOffgridAITextVisemePlan& plan,
    const std::vector<GoldPhoneTiming>&,
    const std::vector<GoldSpeechRegion>&,
    std::vector<double>& out_start_by_index,
    std::vector<double>& out_end_by_index)
{
    std::vector<double> word_start_by_index;
    std::vector<double> word_end_by_index;
    build_runtime_phone_prior_times(
        plan,
        out_start_by_index,
        out_end_by_index,
        word_start_by_index,
        word_end_by_index);
}

static std::vector<LandmarkTarget> build_landmark_targets(
    const FOffgridAITextVisemePlan& plan,
    const std::vector<GoldPhoneTiming>& gold_phones,
    const std::vector<GoldWordTiming>& gold_words,
    const std::vector<GoldBoundaryTiming>& gold_boundaries)
{
    std::vector<LandmarkTarget> targets;
    if (plan.ExpectedPhones.Num() <= 0)
    {
        return targets;
    }

    size_t phone_lookup_size = static_cast<size_t>(plan.ExpectedPhones.Num());
    for (const GoldPhoneTiming& phone : gold_phones)
    {
        if (phone.global_phone_index >= 0)
        {
            phone_lookup_size = std::max(phone_lookup_size, static_cast<size_t>(phone.global_phone_index) + 1);
        }
    }
    std::vector<const GoldPhoneTiming*> by_global_index(phone_lookup_size, nullptr);
    for (const GoldPhoneTiming& phone : gold_phones)
    {
        if (phone.global_phone_index >= 0 && static_cast<size_t>(phone.global_phone_index) < by_global_index.size())
        {
            by_global_index[static_cast<size_t>(phone.global_phone_index)] = &phone;
        }
    }

    size_t word_lookup_size = static_cast<size_t>(plan.WordPhoneBeginIndices.Num());
    for (const GoldWordTiming& word : gold_words)
    {
        if (word.word_index >= 0)
        {
            word_lookup_size = std::max(word_lookup_size, static_cast<size_t>(word.word_index) + 1);
        }
    }
    std::vector<const GoldWordTiming*> words_by_index(word_lookup_size, nullptr);
    for (const GoldWordTiming& word : gold_words)
    {
        if (word.word_index >= 0 && static_cast<size_t>(word.word_index) < words_by_index.size())
        {
            words_by_index[static_cast<size_t>(word.word_index)] = &word;
        }
    }

    std::vector<const GoldBoundaryTiming*> boundaries_by_word_index(
        std::max(
            static_cast<size_t>(plan.WordBoundaryPunctuationAfter.Num()),
            gold_boundaries.size()) + 1,
        nullptr);
    for (const GoldBoundaryTiming& boundary : gold_boundaries)
    {
        if (boundary.word_index >= 0 && static_cast<size_t>(boundary.word_index) < boundaries_by_word_index.size())
        {
            boundaries_by_word_index[static_cast<size_t>(boundary.word_index)] = &boundary;
        }
    }

    std::vector<double> expected_start_by_index;
    std::vector<double> expected_end_by_index;
    std::vector<double> word_start_by_index;
    std::vector<double> word_end_by_index;
    build_runtime_phone_prior_times(
        plan,
        expected_start_by_index,
        expected_end_by_index,
        word_start_by_index,
        word_end_by_index);

    for (int32 i = 0; i < plan.ExpectedPhones.Num(); ++i)
    {
        const FOffgridAIExpectedPhone& expected = plan.ExpectedPhones[i];
        const std::string phone_base = to_std(expected.BasePhone);
        const std::optional<std::string> type = transcript_landmark_type_for_phone_base(phone_base);
        if (!type.has_value())
        {
            continue;
        }

        const GoldPhoneTiming* gold = nullptr;
        if (expected.PhoneIndex >= 0 && static_cast<size_t>(expected.PhoneIndex) < by_global_index.size())
        {
            gold = by_global_index[static_cast<size_t>(expected.PhoneIndex)];
        }
        LandmarkTarget target;
        target.target_index = static_cast<int>(targets.size());
        target.type = *type;
        target.expected_phone_index = i;
        target.global_phone_index = expected.PhoneIndex;
        target.word_index = expected.WordIndex;
        target.speech_region_index = expected.SpeechRegionIndex;
        target.sentence_index = expected.SentenceIndex;
        target.word = to_std(expected.SourceWord);
        target.phone = to_std(expected.Phone);
        target.phone_base = phone_base;
        target.prior_start = expected_start_by_index[static_cast<size_t>(i)];
        target.prior_end = expected_end_by_index[static_cast<size_t>(i)];
        target.prior_center = 0.5 * (target.prior_start + target.prior_end);
        if (gold != nullptr)
        {
            target.gold_start = gold->start;
            target.gold_end = gold->end;
            target.gold_center = 0.5 * (gold->start + gold->end);
            target.speech_region_index = (gold->speech_region_index >= 0)
                ? gold->speech_region_index
                : target.speech_region_index;
            target.has_gold_reference = true;
        }
        targets.push_back(std::move(target));
    }
    const int32 boundary_count = std::min(plan.WordBoundaryPunctuationAfter.Num(), plan.WordBoundaryPauseClassAfter.Num());
    for (int32 word_index = 0; word_index < boundary_count; ++word_index)
    {
        const TCHAR boundary_mark = plan.WordBoundaryPunctuationAfter[word_index];
        const EOffgridAIBoundaryPauseClass pause_class = plan.WordBoundaryPauseClassAfter[word_index];
        if (pause_class == EOffgridAIBoundaryPauseClass::None)
        {
            continue;
        }
        if (word_index + 1 >= static_cast<int32>(word_start_by_index.size()))
        {
            continue;
        }

        const GoldWordTiming* left = (word_index >= 0 && word_index < static_cast<int32>(words_by_index.size()))
            ? words_by_index[static_cast<size_t>(word_index)]
            : nullptr;
        const GoldWordTiming* right = (word_index + 1) >= 0
            && static_cast<size_t>(word_index + 1) < words_by_index.size()
            ? words_by_index[static_cast<size_t>(word_index + 1)]
            : nullptr;
        const double prior_start = word_index >= 0 && static_cast<size_t>(word_index) < word_end_by_index.size()
            ? word_end_by_index[static_cast<size_t>(word_index)]
            : 0.0;
        const double prior_end = (word_index + 1) >= 0 && static_cast<size_t>(word_index + 1) < word_start_by_index.size()
            ? word_start_by_index[static_cast<size_t>(word_index + 1)]
            : prior_start;
        if (prior_end <= prior_start)
        {
            continue;
        }

        LandmarkTarget target;
        target.target_index = static_cast<int>(targets.size());
        target.type = "pause_lull";
        target.word_index = word_index;
        target.next_word_index = word_index + 1;
        target.speech_region_index = left ? left->speech_region_index : 0;
        target.sentence_index = left ? left->sentence_index : (right ? right->sentence_index : 0);
        target.word = left ? left->word : "";
        target.next_word = right ? right->word : "";
        target.boundary_mark = std::string(1, static_cast<char>(boundary_mark));
        target.pause_class = pause_class_name(pause_class);
        target.prior_start = prior_start;
        target.prior_end = prior_end;
        target.prior_center = 0.5 * (prior_start + prior_end);
        target.prior_duration = std::max(0.0, prior_end - prior_start);
        const GoldBoundaryTiming* gold_boundary =
            (word_index >= 0 && static_cast<size_t>(word_index) < boundaries_by_word_index.size())
                ? boundaries_by_word_index[static_cast<size_t>(word_index)]
                : nullptr;
        if (gold_boundary != nullptr && !gold_boundary->pause_class.empty() && gold_boundary->pause_class != "none")
        {
            const double gap_seconds = std::max({
                gold_boundary->direct_word_gap_seconds,
                gold_boundary->phone_gap_seconds,
                gold_boundary->wave_silence_seconds,
                gold_boundary->acoustic_gap_seconds,
                0.0});
            const double center = (left != nullptr && right != nullptr)
                ? 0.5 * (left->end + right->start)
                : target.prior_center;
            target.gold_center = center;
            target.gold_start = center - gap_seconds * 0.5;
            target.gold_end = center + gap_seconds * 0.5;
            target.has_gold_reference = true;
        }
        targets.push_back(std::move(target));
    }
    std::sort(targets.begin(), targets.end(), [](const LandmarkTarget& a, const LandmarkTarget& b) {
        if (a.prior_center != b.prior_center) return a.prior_center < b.prior_center;
        if (a.global_phone_index != b.global_phone_index) return a.global_phone_index < b.global_phone_index;
        if (a.word_index != b.word_index) return a.word_index < b.word_index;
        return a.type < b.type;
    });
    for (size_t i = 0; i < targets.size(); ++i)
    {
        targets[i].target_index = static_cast<int>(i);
    }

    return targets;
}

static std::string prosodic_vowel_family(const std::string& phone_base)
{
    if (phone_base == "UW" || phone_base == "UH" || phone_base == "OW" ||
        phone_base == "OY" || phone_base == "AO") return "round";
    if (phone_base == "IY" || phone_base == "IH" || phone_base == "EY" ||
        phone_base == "EH" || phone_base == "AE") return "front";
    return "open_central";
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

static std::vector<ProsodicPeakObservation> detect_prosodic_peak_observations(
    const FOffgridAITextVisemePlan& plan,
    const std::vector<ProsodicPeakTarget>& targets,
    const TArray<FOffgridAIStreamingAudioFeatureFrame>& frames,
    const TArray<FOffgridAIStreamingSpeechRegion>& speech_regions,
    std::vector<ProsodicPeakObservation>& out_raw_peaks)
{
    struct PeakCandidate
    {
        int frame_index = -1;
        double center = 0.0;
        double decision = 0.0;
        double prominence = 0.0;
    };
    std::vector<PeakCandidate> peaks;
    auto build_envelope = [&](int32 half_width) {
        std::vector<double> result(static_cast<size_t>(frames.Num()), 0.0);
        for (int32 i = 0; i < frames.Num(); ++i)
        {
            double weighted_sum = 0.0;
            double weight_sum = 0.0;
            for (int32 offset = -half_width; offset <= half_width; ++offset)
            {
                const int32 sample_index = i + offset;
                if (!frames.IsValidIndex(sample_index)) continue;
                const double weight = static_cast<double>(half_width + 1 - std::abs(offset));
                weighted_sum += static_cast<double>(frames[sample_index].RMSNorm) * weight;
                weight_sum += weight;
            }
            result[static_cast<size_t>(i)] = weight_sum > 0.0 ? weighted_sum / weight_sum : 0.0;
        }
        return result;
    };
    const std::vector<double> envelope = build_envelope(4);
    const std::vector<double> envelope_slow = build_envelope(6);
    for (int32 i = 14; i + 14 < frames.Num(); ++i)
    {
        auto scale_peak = [&](const std::vector<double>& scale, int32 inner_offset) {
            const double center_rms = scale[static_cast<size_t>(i)];
            double left_floor = center_rms;
            double right_floor = center_rms;
            for (int32 sample_index = i - 14; sample_index <= i - inner_offset; ++sample_index)
                left_floor = std::min(left_floor, scale[static_cast<size_t>(sample_index)]);
            for (int32 sample_index = i + inner_offset; sample_index <= i + 14; ++sample_index)
                right_floor = std::min(right_floor, scale[static_cast<size_t>(sample_index)]);
            const bool local_peak = center_rms >= scale[static_cast<size_t>(i - 1)] &&
                center_rms >= scale[static_cast<size_t>(i + 1)];
            return local_peak ? clamp01(std::min(center_rms - left_floor, center_rms - right_floor) / 0.12) : 0.0;
        };
        const double medium_prominence = scale_peak(envelope, 4);
        const double slow_prominence = scale_peak(envelope_slow, 6);
        const double prominence = std::max(medium_prominence, slow_prominence * 1.05);
        const bool local_peak = prominence > 0.0;
        const FOffgridAIArticulatoryProbabilityField peak_field =
            FOffgridAIOnlinePhoneAligner::BuildArticulatoryProbabilityField(frames[i]);
        const double vowel_support = clamp01(
            static_cast<double>(peak_field.Vowel) * 0.60 +
            static_cast<double>(frames[i].Periodicity) * 0.40);
        if (!local_peak || prominence < 0.16 || frames[i].SpeechEvidence < 0.18f || vowel_support < 0.25)
        {
            continue;
        }
        const PeakCandidate candidate{
            i,
            static_cast<double>(frames[i].AudioBufferCenterSec),
            static_cast<double>(frames[i + 14].AudioBufferEndSec),
            prominence};
        if (!peaks.empty() && candidate.center - peaks.back().center < 0.120)
        {
            if (candidate.prominence > peaks.back().prominence)
            {
                peaks.back() = candidate;
            }
        }
        else
        {
            peaks.push_back(candidate);
        }
    }

    out_raw_peaks.clear();
    for (const PeakCandidate& peak : peaks)
    {
        ProsodicPeakObservation observation;
        observation.frame_index = peak.frame_index;
        observation.peak_sec = peak.center;
        observation.decision_sec = peak.decision;
        observation.prominence = peak.prominence;
        observation.score = peak.prominence;
        out_raw_peaks.push_back(observation);
    }

    std::map<int, double> first_prior_by_region;
    std::map<int, double> last_prior_by_region;
    for (const ProsodicPeakTarget& target : targets)
    {
        auto [first_it, first_inserted] = first_prior_by_region.emplace(target.speech_region_index, target.prior_center);
        if (!first_inserted) first_it->second = std::min(first_it->second, target.prior_center);
        auto [last_it, last_inserted] = last_prior_by_region.emplace(target.speech_region_index, target.prior_center);
        if (!last_inserted) last_it->second = std::max(last_it->second, target.prior_center);
    }

    std::vector<double> expected_centers(targets.size(), 0.0);
    for (size_t target_index = 0; target_index < targets.size(); ++target_index)
    {
        const ProsodicPeakTarget& target = targets[target_index];
        double expected_center = target.prior_center;
        const auto first_it = first_prior_by_region.find(target.speech_region_index);
        const auto last_it = last_prior_by_region.find(target.speech_region_index);
        if (first_it != first_prior_by_region.end() && last_it != last_prior_by_region.end() &&
            speech_regions.IsValidIndex(target.speech_region_index))
        {
            const FOffgridAIStreamingSpeechRegion& region = speech_regions[target.speech_region_index];
            const double prior_span = last_it->second - first_it->second;
            const double observed_span = region.AudioBufferLastSpeechSec - region.AudioBufferStartSec;
            if (region.bEnded && prior_span > 0.05 && observed_span > 0.05)
            {
                const double progress = clamp01((target.prior_center - first_it->second) / prior_span);
                expected_center = region.AudioBufferStartSec + progress * observed_span;
            }
            else
            {
                expected_center += region.AudioBufferStartSec - first_it->second;
            }
        }
        expected_centers[target_index] = expected_center;
    }

    std::vector<std::vector<double>> match_scores(targets.size(), std::vector<double>(peaks.size(), -1.0));
    std::vector<std::vector<double>> family_scores(targets.size(), std::vector<double>(peaks.size(), 0.0));
    for (size_t target_index = 0; target_index < targets.size(); ++target_index)
    {
        const ProsodicPeakTarget& target = targets[target_index];
        for (size_t peak_index = 0; peak_index < peaks.size(); ++peak_index)
        {
            const PeakCandidate& peak = peaks[peak_index];
            const double distance = std::abs(peak.center - expected_centers[target_index]);
            if (distance > 0.55)
            {
                continue;
            }
            const FOffgridAIArticulatoryProbabilityField field =
                FOffgridAIOnlinePhoneAligner::BuildArticulatoryProbabilityField(frames[peak.frame_index]);
            const double family_score = target.vowel_family == "round"
                ? conditioned_landmark_template_score("round", frames, peak.frame_index)
                : clamp01(static_cast<double>(field.Vowel) * 0.65 +
                    static_cast<double>(field.Voiced) * 0.15 +
                    static_cast<double>(frames[peak.frame_index].Periodicity) * 0.20);
            const double proximity = clamp01(1.0 - distance / 0.55);
            const double score = peak.prominence * 0.42 + family_score * 0.38 + proximity * 0.20;
            match_scores[target_index][peak_index] = score;
            family_scores[target_index][peak_index] = family_score;
        }
    }

    std::vector<ProsodicPeakObservation> observations;
    struct SequenceState
    {
        int next_target = 0;
        int last_target = -1;
        int last_peak = -1;
        double local_rate = 1.0;
        double score = 0.0;
        std::vector<std::pair<int, int>> matches;
    };
    for (int32 region_index = 0; region_index < speech_regions.Num(); ++region_index)
    {
        std::vector<int> region_targets;
        std::vector<int> region_peaks;
        for (size_t target_index = 0; target_index < targets.size(); ++target_index)
        {
            if (targets[target_index].speech_region_index == region_index)
            {
                region_targets.push_back(static_cast<int>(target_index));
            }
        }
        const auto& region = speech_regions[region_index];
        for (size_t peak_index = 0; peak_index < peaks.size(); ++peak_index)
        {
            if (peaks[peak_index].center >= region.AudioBufferStartSec - 0.030 &&
                peaks[peak_index].center <= region.AudioBufferEndSec + 0.030)
            {
                region_peaks.push_back(static_cast<int>(peak_index));
            }
        }
        if (region_targets.empty() || region_peaks.empty()) continue;

        std::vector<SequenceState> beam(1);
        for (const int global_peak_index : region_peaks)
        {
            std::vector<SequenceState> next_beam;
            for (const SequenceState& state : beam)
            {
                SequenceState ignored = state;
                ignored.score -= 0.18;
                next_beam.push_back(std::move(ignored));

                for (int skip_count = 0; skip_count <= 2; ++skip_count)
                {
                    const int local_target_index = state.next_target + skip_count;
                    if (local_target_index < 0 || local_target_index >= static_cast<int>(region_targets.size())) continue;
                    const int global_target_index = region_targets[static_cast<size_t>(local_target_index)];
                    const double evidence = match_scores[static_cast<size_t>(global_target_index)][static_cast<size_t>(global_peak_index)];
                    if (evidence < 0.0) continue;

                    double timing_support = clamp01(evidence);
                    double valley_support = 0.50;
                    double updated_rate = state.local_rate;
                    if (state.last_target >= 0 && state.last_peak >= 0)
                    {
                        const int previous_global_target = region_targets[static_cast<size_t>(state.last_target)];
                        const double expected_gap = std::max(0.040,
                            targets[static_cast<size_t>(global_target_index)].prior_center -
                            targets[static_cast<size_t>(previous_global_target)].prior_center);
                        const double observed_gap = std::max(0.010,
                            peaks[static_cast<size_t>(global_peak_index)].center -
                            peaks[static_cast<size_t>(state.last_peak)].center);
                        const double predicted_gap = expected_gap * state.local_rate;
                        timing_support = std::exp(-std::abs(observed_gap - predicted_gap) /
                            std::max(0.090, predicted_gap * 0.55));
                        const double measured_rate = std::max(0.55, std::min(1.65, observed_gap / expected_gap));
                        updated_rate = state.local_rate * 0.65 + measured_rate * 0.35;

                        double valley_floor = std::min(
                            envelope[static_cast<size_t>(peaks[static_cast<size_t>(state.last_peak)].frame_index)],
                            envelope[static_cast<size_t>(peaks[static_cast<size_t>(global_peak_index)].frame_index)]);
                        for (int32 sample_index = peaks[static_cast<size_t>(state.last_peak)].frame_index + 1;
                            sample_index < peaks[static_cast<size_t>(global_peak_index)].frame_index;
                            ++sample_index)
                        {
                            valley_floor = std::min(valley_floor, envelope[static_cast<size_t>(sample_index)]);
                        }
                        const double endpoint_level = std::max(0.02, std::min(
                            envelope[static_cast<size_t>(peaks[static_cast<size_t>(state.last_peak)].frame_index)],
                            envelope[static_cast<size_t>(peaks[static_cast<size_t>(global_peak_index)].frame_index)]));
                        const double valley_depth = clamp01((endpoint_level - valley_floor) / endpoint_level);
                        const int previous_word_index = targets[static_cast<size_t>(previous_global_target)].word_index;
                        const TCHAR boundary_mark = plan.WordBoundaryPunctuationAfter.IsValidIndex(previous_word_index)
                            ? plan.WordBoundaryPunctuationAfter[previous_word_index]
                            : TCHAR(0);
                        if (boundary_mark != TCHAR(0))
                        {
                            const double duration_support = clamp01((observed_gap - 0.090) / 0.180);
                            valley_support = valley_depth * 0.60 + duration_support * 0.40;
                        }
                        else
                        {
                            valley_support = clamp01(0.70 - std::max(0.0, observed_gap - 0.260) * 1.5);
                        }
                    }
                    const ProsodicPeakTarget& target = targets[static_cast<size_t>(global_target_index)];
                    const double stress_support = target.stress == 1
                        ? clamp01(peaks[static_cast<size_t>(global_peak_index)].prominence / 0.55)
                        : clamp01(1.0 - std::abs(peaks[static_cast<size_t>(global_peak_index)].prominence - 0.35));
                    SequenceState matched = state;
                    matched.next_target = local_target_index + 1;
                    matched.last_target = local_target_index;
                    matched.last_peak = global_peak_index;
                    matched.local_rate = updated_rate;
                    matched.score += evidence * 0.34 + timing_support * 0.26 +
                        stress_support * 0.14 + valley_support * 0.20 - skip_count * 0.24;
                    matched.matches.emplace_back(global_target_index, global_peak_index);
                    next_beam.push_back(std::move(matched));
                }
            }
            std::sort(next_beam.begin(), next_beam.end(), [](const auto& a, const auto& b) {
                if (a.score != b.score) return a.score > b.score;
                return a.next_target > b.next_target;
            });
            std::vector<SequenceState> pruned;
            for (SequenceState& candidate : next_beam)
            {
                const bool duplicate = std::any_of(pruned.begin(), pruned.end(), [&](const SequenceState& kept) {
                    return kept.next_target == candidate.next_target && kept.last_target == candidate.last_target;
                });
                if (!duplicate) pruned.push_back(std::move(candidate));
                if (pruned.size() >= 8) break;
            }
            beam = std::move(pruned);
        }
        if (beam.empty()) continue;
        const SequenceState& best = beam.front();
        for (const auto& [global_target_index, global_peak_index] : best.matches)
        {
            const PeakCandidate& peak = peaks[static_cast<size_t>(global_peak_index)];
            ProsodicPeakObservation observation;
            observation.frame_index = peak.frame_index;
            observation.target_index = targets[static_cast<size_t>(global_target_index)].target_index;
            observation.peak_sec = peak.center;
            observation.decision_sec = peak.decision;
            observation.prominence = peak.prominence;
            observation.family_score = family_scores[static_cast<size_t>(global_target_index)][static_cast<size_t>(global_peak_index)];
            observation.score = match_scores[static_cast<size_t>(global_target_index)][static_cast<size_t>(global_peak_index)];
            observations.push_back(observation);
        }
    }
    std::sort(observations.begin(), observations.end(), [](const auto& a, const auto& b) { return a.peak_sec < b.peak_sec; });
    return observations;
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

    std::vector<std::pair<double, double>> gold_intervals;
    for (const ProsodicPeakTarget& target : targets)
    {
        if (target.has_gold) gold_intervals.emplace_back(target.gold_start, target.gold_end);
    }
    std::vector<bool> used_gold(gold_intervals.size(), false);
    std::vector<double> raw_errors;
    report.raw_observation_count = static_cast<int>(raw_peaks.size());
    for (const ProsodicPeakObservation& peak : raw_peaks)
    {
        int best_index = -1;
        double best_error = 0.100001;
        for (size_t target_index = 0; target_index < gold_intervals.size(); ++target_index)
        {
            if (used_gold[target_index]) continue;
            const auto [start, end] = gold_intervals[target_index];
            const double error = peak.peak_sec < start
                ? start - peak.peak_sec
                : (peak.peak_sec > end ? peak.peak_sec - end : 0.0);
            if (error < best_error)
            {
                best_error = error;
                best_index = static_cast<int>(target_index);
            }
        }
        if (best_index >= 0)
        {
            used_gold[static_cast<size_t>(best_index)] = true;
            ++report.raw_matched_count;
            raw_errors.push_back(best_error * 1000.0);
        }
    }
    report.raw_precision = report.raw_observation_count > 0
        ? static_cast<double>(report.raw_matched_count) / report.raw_observation_count : 0.0;
    report.raw_recall = report.target_count > 0
        ? static_cast<double>(report.raw_matched_count) / report.target_count : 0.0;
    report.raw_mean_abs_error_ms = mean_ms(raw_errors);
    return report;
}

static std::vector<ProsodicPeakObservation> runtime_syllable_assignment_observations(
    const std::vector<ProsodicPeakTarget>& targets,
    const TArray<FOffgridAIRuntimeBoundaryDiagnosticRow>& rows)
{
    std::map<int, int> target_by_phone_index;
    for (const ProsodicPeakTarget& target : targets)
    {
        target_by_phone_index[target.expected_phone_index] = target.target_index;
    }
    std::vector<ProsodicPeakObservation> observations;
    int32 LastAssignmentCount = 0;
    for (const FOffgridAIRuntimeBoundaryDiagnosticRow& row : rows)
    {
        if (row.SyllableAssignmentCount <= LastAssignmentCount) continue;
        LastAssignmentCount = row.SyllableAssignmentCount;
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
            observation.score = row.SyllableAssignmentConfidence;
            observation.family_score = row.SyllableAssignmentMargin;
            observations.push_back(observation);
        };
        Emit(row.SyllableLastAssignedPhoneIndex, row.SyllableObservedAudioSec);
    }
    return observations;
}

static std::string prosodic_peak_targets_csv(const std::vector<ProsodicPeakTarget>& targets)
{
    std::ostringstream out;
    out << "target_index,expected_phone_index,global_phone_index,word_index,speech_region_index,stress,word,phone,vowel_family,prior_center,gold_start,gold_end,gold_center,has_gold\n";
    for (const auto& target : targets)
    {
        out << target.target_index << ',' << target.expected_phone_index << ',' << target.global_phone_index << ','
            << target.word_index << ',' << target.speech_region_index << ',' << target.stress << ','
            << target.word << ',' << target.phone << ',' << target.vowel_family << ','
            << target.prior_center << ',' << target.gold_start << ',' << target.gold_end << ','
            << target.gold_center << ',' << (target.has_gold ? 1 : 0) << '\n';
    }
    return out.str();
}

static std::string prosodic_peak_observations_csv(const std::vector<ProsodicPeakObservation>& observations)
{
    std::ostringstream out;
    out << "frame_index,target_index,peak_sec,decision_sec,prominence,family_score,score,error_ms,matched\n";
    for (const auto& observation : observations)
    {
        out << observation.frame_index << ',' << observation.target_index << ',' << observation.peak_sec << ','
            << observation.decision_sec << ',' << observation.prominence << ',' << observation.family_score << ','
            << observation.score << ',' << observation.error_ms << ',' << (observation.matched ? 1 : 0) << '\n';
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
        << "  \"raw_matched_count\": " << report.raw_matched_count << ",\n"
        << "  \"raw_precision\": " << report.raw_precision << ",\n"
        << "  \"raw_recall\": " << report.raw_recall << ",\n"
        << "  \"raw_mean_abs_error_ms\": " << report.raw_mean_abs_error_ms << "\n}\n";
    return out.str();
}

static std::string strict_punctuation_alignment_json(
    const std::vector<LandmarkTarget>& targets,
    const std::vector<LandmarkObservation>& observations)
{
    struct Counts { int targets = 0; int closes = 0; int close_matches = 0; int resumes = 0; int resume_matches = 0; };
    std::map<std::string, Counts> by_class;
    const double tolerance_sec = 0.100;
    for (size_t target_index = 0; target_index < targets.size(); ++target_index)
    {
        const auto& target = targets[target_index];
        if (target.type != "pause_lull" || !target.has_gold_reference) continue;
        Counts& counts = by_class[target.pause_class.empty() ? "unknown" : target.pause_class];
        ++counts.targets;
        for (const auto& observation : observations)
        {
            if (observation.target_index != static_cast<int>(target_index)) continue;
            if (observation.type == "pause_close" && observation.source == "boundary_close")
            {
                ++counts.closes;
                if (std::abs(observation.center - target.gold_start) <= tolerance_sec) ++counts.close_matches;
            }
            else if (observation.type == "resume" && observation.source == "boundary_resume")
            {
                ++counts.resumes;
                if (std::abs(observation.center - target.gold_end) <= tolerance_sec) ++counts.resume_matches;
            }
        }
    }
    Counts total;
    for (const auto& [_, counts] : by_class)
    {
        total.targets += counts.targets;
        total.closes += counts.closes;
        total.close_matches += counts.close_matches;
        total.resumes += counts.resumes;
        total.resume_matches += counts.resume_matches;
    }
    auto ratio = [](int numerator, int denominator) { return denominator > 0 ? static_cast<double>(numerator) / denominator : 0.0; };
    std::ostringstream out;
    out << std::fixed << std::setprecision(6)
        << "{\n  \"available\": true,\n"
        << "  \"target_count\": " << total.targets << ",\n"
        << "  \"close_observation_count\": " << total.closes << ",\n"
        << "  \"close_match_count\": " << total.close_matches << ",\n"
        << "  \"close_precision\": " << ratio(total.close_matches, total.closes) << ",\n"
        << "  \"close_recall\": " << ratio(total.close_matches, total.targets) << ",\n"
        << "  \"resume_observation_count\": " << total.resumes << ",\n"
        << "  \"resume_match_count\": " << total.resume_matches << ",\n"
        << "  \"resume_precision\": " << ratio(total.resume_matches, total.resumes) << ",\n"
        << "  \"resume_recall\": " << ratio(total.resume_matches, total.targets) << ",\n"
        << "  \"classes\": {";
    bool first = true;
    for (const auto& [name, counts] : by_class)
    {
        if (!first) out << ',';
        first = false;
        out << "\n    \"" << name << "\": {\"target_count\": " << counts.targets
            << ", \"close_precision\": " << ratio(counts.close_matches, counts.closes)
            << ", \"close_recall\": " << ratio(counts.close_matches, counts.targets)
            << ", \"resume_precision\": " << ratio(counts.resume_matches, counts.resumes)
            << ", \"resume_recall\": " << ratio(counts.resume_matches, counts.targets) << '}';
    }
    out << "\n  }\n}\n";
    return out.str();
}

static std::string strict_punctuation_alignment_csv(
    const std::vector<LandmarkTarget>& targets,
    const std::vector<LandmarkObservation>& observations)
{
    std::ostringstream out;
    out << "target_index,word_index,boundary_code,pause_class,gold_close,gold_resume,observed_close,observed_resume,close_error_ms,resume_error_ms,close_matched,resume_matched\n";
    const double tolerance_sec = 0.100;
    for (size_t target_index = 0; target_index < targets.size(); ++target_index)
    {
        const auto& target = targets[target_index];
        if (target.type != "pause_lull" || !target.has_gold_reference) continue;
        const LandmarkObservation* close = nullptr;
        const LandmarkObservation* resume = nullptr;
        for (const auto& observation : observations)
        {
            if (observation.target_index != static_cast<int>(target_index)) continue;
            if (observation.type == "pause_close" && observation.source == "boundary_close") close = &observation;
            if (observation.type == "resume" && observation.source == "boundary_resume") resume = &observation;
        }
        const double close_error = close ? (close->center - target.gold_start) * 1000.0 : 0.0;
        const double resume_error = resume ? (resume->center - target.gold_end) * 1000.0 : 0.0;
        const bool close_matched = close && std::abs(close->center - target.gold_start) <= tolerance_sec;
        const bool resume_matched = resume && std::abs(resume->center - target.gold_end) <= tolerance_sec;
        const int boundary_code = target.boundary_mark.empty() ? 0 : static_cast<unsigned char>(target.boundary_mark[0]);
        out << target_index << ',' << target.word_index << ',' << boundary_code << ',' << target.pause_class << ','
            << target.gold_start << ',' << target.gold_end << ',' << (close ? close->center : -1.0) << ','
            << (resume ? resume->center : -1.0) << ',' << close_error << ',' << resume_error << ','
            << (close_matched ? 1 : 0) << ',' << (resume_matched ? 1 : 0) << '\n';
    }
    return out.str();
}

static std::string transcript_landmarks_csv(const std::vector<LandmarkTarget>& targets)
{
    std::ostringstream out;
    out << "target_index,type,expected_phone_index,global_phone_index,word_index,next_word_index,speech_region_index,sentence_index,word,next_word,phone,phone_base,boundary_mark,has_gold_reference,gold_start,gold_end,gold_center,prior_start,prior_end,prior_center\n";
    out << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < targets.size(); ++i)
    {
        const auto& target = targets[i];
        out << i << ','
            << target.type << ','
            << target.expected_phone_index << ','
            << target.global_phone_index << ','
            << target.word_index << ','
            << target.next_word_index << ','
            << target.speech_region_index << ','
            << target.sentence_index << ','
            << target.word << ','
            << target.next_word << ','
            << target.phone << ','
            << target.phone_base << ','
            << target.boundary_mark << ','
            << (target.has_gold_reference ? 1 : 0) << ','
            << target.gold_start << ','
            << target.gold_end << ','
            << target.gold_center << ','
            << target.prior_start << ','
            << target.prior_end << ','
            << target.prior_center << '\n';
    }
    return out.str();
}


static std::string audio_landmark_frames_csv(const TArray<FOffgridAIStreamingAudioFeatureFrame>& frames)
{
    std::ostringstream out;
    out << "frame_index,start,end,center,mbp,fv,w,chjjsh,round,comma_lull,top_landmark,top_score\n";
    out << std::fixed << std::setprecision(6);

    for (int32 i = 0; i < frames.Num(); ++i)
    {
        const auto& frame = frames[i];
        const FOffgridAIArticulatoryProbabilityField field = FOffgridAIOnlinePhoneAligner::BuildArticulatoryProbabilityField(frame);
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
            << top_type << ','
            << top_score << '\n';
    }

    return out.str();
}

static std::vector<LandmarkObservation> detect_transcript_conditioned_landmark_observations(
    const TArray<FOffgridAIStreamingAudioFeatureFrame>& frames,
    const TArray<FOffgridAIStreamingSpeechRegion>& speech_regions,
    const TArray<FOffgridAIStreamingSpeechGapCandidate>& gaps,
    const TArray<FOffgridAIStreamingSoftLullCandidate>& soft_lulls,
    const std::vector<LandmarkTarget>& targets)
{
    auto detect_pause_boundary_observations = [&]() -> std::vector<LandmarkObservation> {
        struct PauseInterval
        {
            int32 close_frame_index = -1;
            int32 reopen_frame_index = -1;
            double start = 0.0;
            double end = 0.0;
            double center = 0.0;
            double score = 0.0;
            double confidence = 0.0;
            double close_decision_sec = 0.0;
            double resume_decision_sec = 0.0;
            bool advisory_soft_lull = false;
            bool bridged_gap = false;
            bool unbridged_gap = false;
            double duration_sec = 0.0;
        };

        std::vector<PauseInterval> intervals;
        auto frame_center = [&](int32 index) -> double {
            return frames.IsValidIndex(index)
                ? static_cast<double>(frames[index].AudioBufferCenterSec)
                : 0.0;
        };
        auto find_close_frame_index = [&](double start_sec, double end_sec) -> int32 {
            int32 best = INDEX_NONE;
            for (int32 i = 0; i < frames.Num(); ++i)
            {
                const auto& frame = frames[i];
                const double frame_center = static_cast<double>(frame.AudioBufferCenterSec);
                if (frame_center + 0.020 < start_sec || frame_center - 0.020 > end_sec)
                {
                    continue;
                }
                if (frame.bFrameClosedSpeechRegion)
                {
                    return i;
                }
                if (best == INDEX_NONE
                    && (frame.bStrongQuiet || (!frame.bInSpeechAfterFrame && frame.SpeechEvidence <= 0.24f)))
                {
                    best = i;
                }
            }
            return best;
        };
        auto find_resume_frame_index = [&](double end_sec) -> int32 {
            int32 best = INDEX_NONE;
            for (int32 i = 0; i < frames.Num(); ++i)
            {
                const auto& frame = frames[i];
                const double frame_center = static_cast<double>(frame.AudioBufferCenterSec);
                if (frame_center + 0.030 < end_sec || frame_center - 0.080 > end_sec)
                {
                    continue;
                }
                if (frame.bFrameStartedSpeechRegion)
                {
                    return i;
                }
                if (best == INDEX_NONE
                    && (frame.bStrongOnsetAnchor || (!frame.bInSpeechBeforeFrame && frame.bInSpeechAfterFrame)))
                {
                    best = i;
                }
            }
            return best;
        };

        for (const FOffgridAIStreamingSpeechGapCandidate& gap : gaps)
        {
            if (gap.GapDurationSec <= 0.0f)
            {
                continue;
            }

            const double mean_evidence = gap.GapFrameCount > 0
                ? static_cast<double>(gap.GapEvidenceSum) / static_cast<double>(gap.GapFrameCount)
                : static_cast<double>(gap.QuietEvidence);
            const double mean_rms = gap.GapFrameCount > 0
                ? static_cast<double>(gap.GapRMSNormSum) / static_cast<double>(gap.GapFrameCount)
                : static_cast<double>(gap.QuietRMSNorm);
            const double quiet_support = clamp01((0.26 - static_cast<double>(gap.QuietEvidence)) / 0.26);
            const double mean_quiet_support = clamp01((0.24 - mean_evidence) / 0.24);
            const double quiet_floor = clamp01((0.12 - static_cast<double>(gap.QuietRMSNorm)) / 0.12);
            const double mean_floor = clamp01((0.14 - mean_rms) / 0.14);
            const double duration_support = clamp01((static_cast<double>(gap.GapDurationSec) - 0.030) / 0.130);
            const double reopen_support = clamp01((static_cast<double>(gap.ReopenEvidence) - 0.20) / 0.30);
            const double reopen_flux_support = clamp01((static_cast<double>(gap.ReopenFlux) - 0.04) / 0.20);
            const double low_evidence_ratio = gap.GapFrameCount > 0
                ? static_cast<double>(gap.LowEvidenceFrameCount) / static_cast<double>(gap.GapFrameCount)
                : 0.0;
            const double strong_quiet_ratio = gap.GapFrameCount > 0
                ? static_cast<double>(gap.StrongQuietFrameCount) / static_cast<double>(gap.GapFrameCount)
                : 0.0;
            const double class_bonus =
                gap.DecisionClass == FName(TEXT("bridge_window_expired_split")) ? 0.10 :
                gap.DecisionClass == FName(TEXT("short_isolated_restart_split")) ? 0.08 :
                gap.DecisionClass == FName(TEXT("collapsed_rhetorical_split")) ? 0.08 :
                gap.DecisionClass == FName(TEXT("isolated_pulse_split")) ? 0.06 :
                gap.bBridged ? 0.03 : 0.0;

            const double close_confidence = clamp01(
                quiet_support * 0.24 +
                mean_quiet_support * 0.18 +
                quiet_floor * 0.18 +
                mean_floor * 0.10 +
                duration_support * 0.12 +
                low_evidence_ratio * 0.10 +
                strong_quiet_ratio * 0.05 +
                (gap.bStrongQuietClose ? 0.08 : 0.0) +
                class_bonus);
            const double resume_confidence = clamp01(
                reopen_support * 0.34 +
                reopen_flux_support * 0.20 +
                duration_support * 0.10 +
                quiet_support * 0.10 +
                (gap.bStrongOnsetReopen ? 0.16 : 0.0) +
                class_bonus);
            const double interval_confidence = clamp01(close_confidence * 0.52 + resume_confidence * 0.48);
            const bool keep_as_close_heavy_hard_gap =
                static_cast<double>(gap.GapDurationSec) >= 0.18
                && close_confidence >= 0.60;
            if (interval_confidence < 0.42 && !keep_as_close_heavy_hard_gap)
            {
                continue;
            }

            PauseInterval interval;
            interval.start = static_cast<double>(gap.GapStartSec);
            interval.end = static_cast<double>(gap.GapEndSec);
            interval.center = 0.5 * (interval.start + interval.end);
            interval.duration_sec = interval.end - interval.start;
            interval.score = interval_confidence;
            interval.confidence = interval_confidence;
            interval.bridged_gap = gap.bBridged;
            interval.unbridged_gap = !gap.bBridged;
            interval.close_frame_index = find_close_frame_index(interval.start, interval.end);
            interval.reopen_frame_index = find_resume_frame_index(interval.end);
            const double close_mark_sec = !gap.bBridged
                ? (interval.start + std::min(0.060, std::max(0.020, static_cast<double>(gap.GapDurationSec) * 0.20)))
                : interval.start;
            interval.close_decision_sec = interval.close_frame_index != INDEX_NONE
                ? static_cast<double>(frames[interval.close_frame_index].AudioBufferEndSec)
                : close_mark_sec;
            interval.start = close_mark_sec;
            interval.resume_decision_sec = interval.reopen_frame_index != INDEX_NONE
                ? static_cast<double>(frames[interval.reopen_frame_index].AudioBufferEndSec)
                : interval.end + 0.010;
            intervals.push_back(interval);
        }

        for (const FOffgridAIStreamingSoftLullCandidate& lull : soft_lulls)
        {
            if (lull.LullDurationSec < 0.030f || lull.FrameCount <= 0)
            {
                continue;
            }
            if (lull.ReopenEvidence <= 0.0f
                && lull.ReopenFlux <= 0.0f
                && !lull.bStrongOnsetReopen)
            {
                continue;
            }

            // True detector gaps already carry richer endpoint diagnostics.
            const bool overlaps_gap = std::any_of(gaps.begin(), gaps.end(), [&](const auto& gap) {
                return lull.LullStartSec < gap.GapEndSec + 0.025f
                    && lull.LullEndSec > gap.GapStartSec - 0.025f;
            });
            if (overlaps_gap)
            {
                continue;
            }

            const double mean_relative_rms = static_cast<double>(lull.RelativeRMSSum) / lull.FrameCount;
            const double mean_evidence = static_cast<double>(lull.EvidenceSum) / lull.FrameCount;
            const double depth = clamp01((0.24 - mean_relative_rms) / 0.20);
            const double floor = clamp01((0.18 - static_cast<double>(lull.RelativeRMSMin)) / 0.16);
            const double quiet = clamp01((0.16 - mean_evidence) / 0.14);
            const double duration = clamp01((static_cast<double>(lull.LullDurationSec) - 0.025) / 0.095);
            const double reopen = clamp01((static_cast<double>(lull.ReopenEvidence) - 0.12) / 0.30);
            const bool keep_soft_lull =
                lull.LullDurationSec >= 0.050f
                && (lull.bStrongOnsetReopen || lull.ReopenEvidence >= 0.18f);
            const double confidence = clamp01(
                depth * 0.20 + floor * 0.18 + quiet * 0.10 + duration * 0.28 +
                reopen * 0.16 + (lull.bStrongOnsetReopen ? 0.12 : 0.0));
            if (confidence < 0.48 || !keep_soft_lull)
            {
                continue;
            }

            PauseInterval interval;
            interval.start = lull.LullStartSec;
            interval.end = lull.LullEndSec;
            interval.center = 0.5 * (interval.start + interval.end);
            interval.duration_sec = interval.end - interval.start;
            interval.score = confidence;
            interval.confidence = confidence;
            interval.close_frame_index = find_close_frame_index(interval.start, interval.end);
            interval.reopen_frame_index = find_resume_frame_index(interval.end);
            interval.close_decision_sec = interval.start + std::min(0.050, static_cast<double>(lull.LullDurationSec));
            interval.resume_decision_sec = interval.end + 0.010;
            interval.advisory_soft_lull = true;
            intervals.push_back(interval);
        }

        std::vector<int> pause_target_indices;
        pause_target_indices.reserve(targets.size());
        for (size_t target_index = 0; target_index < targets.size(); ++target_index)
        {
            if (targets[target_index].type == "pause_lull")
            {
                pause_target_indices.push_back(static_cast<int>(target_index));
            }
        }

        std::sort(intervals.begin(), intervals.end(), [](const PauseInterval& a, const PauseInterval& b) {
            if (a.center != b.center) return a.center < b.center;
            return a.start < b.start;
        });

        std::vector<LandmarkObservation> events;
        const int target_count = static_cast<int>(pause_target_indices.size());
        const int interval_count = static_cast<int>(intervals.size());
        if (target_count <= 0 || interval_count <= 0)
        {
            return events;
        }

        std::vector<int> matched_interval_for_target(static_cast<size_t>(target_count), -1);
        int interval_cursor = 0;
        double resume_rebase = speech_regions.Num() > 0
            ? static_cast<double>(speech_regions[0].AudioBufferStartSec)
            : 0.0;
        for (int target_pos = 0; target_pos < target_count; ++target_pos)
        {
            const LandmarkTarget& target = targets[static_cast<size_t>(pause_target_indices[static_cast<size_t>(target_pos)])];
            const bool hard_fence = target.pause_class == "hard_break_pause";
            const double predicted_center = target.prior_center + resume_rebase;
            const double search_start = predicted_center - (hard_fence ? 0.16 : 0.12);
            const double search_end = hard_fence
                ? std::numeric_limits<double>::infinity()
                : predicted_center + 0.60;
            int best_interval = -1;
            double best_rank = -1.0e18;
            for (int interval_pos = interval_cursor; interval_pos < interval_count; ++interval_pos)
            {
                const PauseInterval& interval = intervals[static_cast<size_t>(interval_pos)];
                if (interval.center < search_start) continue;
                if (interval.center > search_end) break;
                if (hard_fence && interval.advisory_soft_lull
                    && (interval.confidence < 0.68 || interval.duration_sec < 0.05))
                    continue;
                if (!hard_fence && !interval.advisory_soft_lull && interval.unbridged_gap && interval.duration_sec < 0.18)
                    continue;
                const double distance_penalty = std::min(0.50, std::abs(interval.center - predicted_center) * 0.35);
                const double sustained_pause_bonus = clamp01((interval.duration_sec - 0.04) / 0.20) * 0.10;
                const double rank = interval.confidence + interval.score * 0.20 - distance_penalty +
                    sustained_pause_bonus +
                    (hard_fence && interval.unbridged_gap ? 0.16 : 0.0) +
                    (hard_fence && interval.advisory_soft_lull ? -0.10 : 0.0) +
                    (!hard_fence && interval.advisory_soft_lull ? 0.12 : 0.0);
                if (rank > best_rank)
                {
                    best_rank = rank;
                    best_interval = interval_pos;
                }
            }
            if (best_interval < 0) continue;
            matched_interval_for_target[static_cast<size_t>(target_pos)] = best_interval;
            interval_cursor = best_interval + 1;
            const PauseInterval& matched = intervals[static_cast<size_t>(best_interval)];
            resume_rebase = matched.end - target.prior_end;
        }

        for (int target_pos_iter = 0; target_pos_iter < target_count; ++target_pos_iter)
        {
            const int best_interval_index = matched_interval_for_target[static_cast<size_t>(target_pos_iter)];
            if (best_interval_index < 0)
            {
                continue;
            }

            const int target_index = pause_target_indices[static_cast<size_t>(target_pos_iter)];
            const PauseInterval& interval = intervals[static_cast<size_t>(best_interval_index)];

            LandmarkObservation close_obs;
            close_obs.target_index = target_index;
            close_obs.type = "pause_close";
            close_obs.frame_index = interval.close_frame_index;
            close_obs.start = interval.start;
            close_obs.end = interval.start;
            close_obs.center = interval.start;
            close_obs.score = interval.score;
            close_obs.confidence = interval.confidence;
            close_obs.decision_sec = interval.close_decision_sec;
            close_obs.top_class = "pause_close";
            close_obs.source = "boundary_close";
            events.push_back(std::move(close_obs));

            LandmarkObservation pause_obs;
            pause_obs.target_index = target_index;
            pause_obs.type = "pause_lull";
            pause_obs.frame_index = interval.reopen_frame_index >= 0 ? interval.reopen_frame_index : interval.close_frame_index;
            pause_obs.start = interval.start;
            pause_obs.end = interval.end;
            pause_obs.center = interval.center;
            pause_obs.score = interval.score;
            pause_obs.confidence = interval.confidence;
            pause_obs.decision_sec = interval.resume_decision_sec;
            pause_obs.top_class = "pause_lull";
            pause_obs.source = "boundary_interval";
            events.push_back(std::move(pause_obs));

            LandmarkObservation resume_obs;
            resume_obs.target_index = target_index;
            resume_obs.type = "resume";
            resume_obs.frame_index = interval.reopen_frame_index;
            resume_obs.start = interval.end;
            resume_obs.end = interval.end;
            resume_obs.center = interval.end;
            resume_obs.score = interval.score;
            resume_obs.confidence = interval.confidence;
            resume_obs.decision_sec = interval.resume_decision_sec;
            resume_obs.top_class = "resume";
            resume_obs.source = "boundary_resume";
            events.push_back(std::move(resume_obs));
        }

        std::sort(events.begin(), events.end(), [](const LandmarkObservation& a, const LandmarkObservation& b) {
            if (a.center != b.center) return a.center < b.center;
            return a.type < b.type;
        });
        return events;
    };

    struct Candidate
    {
        int32 frame_index = -1;
        double center = 0.0;
        double score = 0.0;
    };

    std::vector<LandmarkObservation> observations;
    if (frames.Num() <= 0 || targets.empty())
    {
        return observations;
    }

    auto local_peak_for_target = [&](const LandmarkTarget& target, double expected_center, double half_window) -> double {
        double best = 0.0;
        for (int32 sample_index = 0; sample_index < frames.Num(); ++sample_index)
        {
            const double sample_center = static_cast<double>(frames[sample_index].AudioBufferCenterSec);
            if (std::abs(sample_center - expected_center) > half_window)
            {
                continue;
            }
            best = std::max(best, conditioned_landmark_target_score(target, frames, sample_index));
        }
        return best;
    };

    std::vector<double> aligned_prior_centers(targets.size(), 0.0);
    std::map<int, double> first_prior_by_region;
    std::map<int, double> last_prior_by_region;
    for (const LandmarkTarget& target : targets)
    {
        if (!is_pause_lull_type(target.type))
        {
            auto [it, inserted] = first_prior_by_region.emplace(target.speech_region_index, target.prior_center);
            if (!inserted)
            {
                it->second = std::min(it->second, target.prior_center);
            }
            auto [last_it, last_inserted] = last_prior_by_region.emplace(target.speech_region_index, target.prior_center);
            if (!last_inserted)
            {
                last_it->second = std::max(last_it->second, target.prior_center);
            }
        }
    }
    for (size_t target_index = 0; target_index < targets.size(); ++target_index)
    {
        const LandmarkTarget& target = targets[target_index];
        double center = target.prior_center;
        const auto first_it = first_prior_by_region.find(target.speech_region_index);
        if (first_it != first_prior_by_region.end() && speech_regions.IsValidIndex(target.speech_region_index))
        {
            const FOffgridAIStreamingSpeechRegion& region = speech_regions[target.speech_region_index];
            const auto last_it = last_prior_by_region.find(target.speech_region_index);
            const double prior_span = last_it != last_prior_by_region.end() ? last_it->second - first_it->second : 0.0;
            const double observed_span = static_cast<double>(region.AudioBufferLastSpeechSec - region.AudioBufferStartSec);
            if (region.bEnded && prior_span > 0.050 && observed_span > 0.050)
            {
                const double progress = clamp01((target.prior_center - first_it->second) / prior_span);
                center = static_cast<double>(region.AudioBufferStartSec) + progress * observed_span;
            }
            else
            {
                center += static_cast<double>(region.AudioBufferStartSec) - first_it->second;
            }
        }
        aligned_prior_centers[target_index] = center;
    }

    std::vector<std::vector<Candidate>> candidates_by_target(targets.size());
    for (size_t target_index = 0; target_index < targets.size(); ++target_index)
    {
        const LandmarkTarget& target = targets[target_index];
        if (is_pause_lull_type(target.type))
        {
            candidates_by_target[target_index].push_back(Candidate{});
            continue;
        }
        const double half_window = landmark_conditioned_half_window_seconds(target.type);
        const double aligned_prior_center = aligned_prior_centers[target_index];
        double search_start = aligned_prior_center - half_window;
        double search_end = aligned_prior_center + half_window;
        // Adjacent targets may be displaced together by speaking-rate drift.
        // Do not partition their searches at stale prior midpoints; monotonic DP
        // below owns target ordering and resolves competing candidates.

        std::vector<Candidate> local_candidates;
        const double threshold = std::max(0.18, landmark_conditioned_detection_threshold(target.type) + 0.02);

        for (int32 i = 0; i < frames.Num(); ++i)
        {
            const auto& frame = frames[i];
            const double center = static_cast<double>(frame.AudioBufferCenterSec);
            if (center < search_start || center > search_end)
            {
                continue;
            }

            const double distance_penalty = (half_window > 0.0)
                ? (std::abs(center - aligned_prior_center) / half_window) * landmark_conditioned_distance_penalty_scale(target.type)
                : 0.0;
            const double base_score = conditioned_landmark_target_score(target, frames, i);
            double neighbor_bonus = 0.0;
            if (target_index > 0)
            {
                const LandmarkTarget& prev_target = targets[target_index - 1];
                const double prior_gap = std::max(0.020, target.prior_center - prev_target.prior_center);
                const double expected_prev_center = center - prior_gap;
                const double prev_half_window = std::min(0.050, std::max(0.020, landmark_conditioned_half_window_seconds(prev_target.type) * 0.5));
                const double prev_peak = local_peak_for_target(prev_target, expected_prev_center, prev_half_window);
                const double prev_threshold = std::max(0.10, landmark_conditioned_detection_threshold(prev_target.type) - 0.08);
                neighbor_bonus += 0.10 * clamp01((prev_peak - prev_threshold) / 0.20);
            }
            if (target_index + 1 < targets.size())
            {
                const LandmarkTarget& next_target = targets[target_index + 1];
                const double prior_gap = std::max(0.020, next_target.prior_center - target.prior_center);
                const double expected_next_center = center + prior_gap;
                const double next_half_window = std::min(0.050, std::max(0.020, landmark_conditioned_half_window_seconds(next_target.type) * 0.5));
                const double next_peak = local_peak_for_target(next_target, expected_next_center, next_half_window);
                const double next_threshold = std::max(0.10, landmark_conditioned_detection_threshold(next_target.type) - 0.08);
                neighbor_bonus += 0.10 * clamp01((next_peak - next_threshold) / 0.20);
            }
            double x_lull_y_bonus = 0.0;
            if (target.type == "comma_lull" && target_index > 0 && target_index + 1 < targets.size())
            {
                const LandmarkTarget& prev_target = targets[target_index - 1];
                const LandmarkTarget& next_target = targets[target_index + 1];
                const double prev_gap = std::max(0.020, target.prior_center - prev_target.prior_center);
                const double next_gap = std::max(0.020, next_target.prior_center - target.prior_center);
                const double expected_prev_center = center - prev_gap;
                const double expected_next_center = center + next_gap;
                const double prev_peak = local_peak_for_target(
                    prev_target,
                    expected_prev_center,
                    std::min(0.060, std::max(0.020, landmark_conditioned_half_window_seconds(prev_target.type) * 0.6)));
                const double next_peak = local_peak_for_target(
                    next_target,
                    expected_next_center,
                    std::min(0.060, std::max(0.020, landmark_conditioned_half_window_seconds(next_target.type) * 0.6)));
                const double prev_threshold = std::max(0.10, landmark_conditioned_detection_threshold(prev_target.type) - 0.08);
                const double next_threshold = std::max(0.10, landmark_conditioned_detection_threshold(next_target.type) - 0.08);
                const double prev_support = clamp01((prev_peak - prev_threshold) / 0.18);
                const double next_support = clamp01((next_peak - next_threshold) / 0.18);
                x_lull_y_bonus = 0.16 * std::min(prev_support, next_support);
            }
            const double competitor_score = conditioned_landmark_competitor_score(target, frames, i);
            const double family_margin = base_score - competitor_score;
            const double local_stability = (
                conditioned_landmark_target_score(target, frames, i - 1) * 0.25 +
                base_score * 0.50 +
                conditioned_landmark_target_score(target, frames, i + 1) * 0.25);
            const double score = base_score - distance_penalty + neighbor_bonus + x_lull_y_bonus;
            const double required_margin =
                (target.type == "chjjsh" || target.type == "fv") ? 0.03 : 0.0;
            if (score < threshold || family_margin < required_margin || local_stability < (threshold - 0.05))
            {
                continue;
            }

            const double prev_score = frames.IsValidIndex(i - 1)
                ? (conditioned_landmark_target_score(target, frames, i - 1) -
                   ((half_window > 0.0)
                ? (std::abs(static_cast<double>(frames[i - 1].AudioBufferCenterSec) - aligned_prior_center) / half_window) *
                            landmark_conditioned_distance_penalty_scale(target.type)
                        : 0.0))
                : score;
            const double next_score = frames.IsValidIndex(i + 1)
                ? (conditioned_landmark_target_score(target, frames, i + 1) -
                   ((half_window > 0.0)
                ? (std::abs(static_cast<double>(frames[i + 1].AudioBufferCenterSec) - aligned_prior_center) / half_window) *
                            landmark_conditioned_distance_penalty_scale(target.type)
                        : 0.0))
                : score;
            if (score + 1e-6 < prev_score || score + 1e-6 < next_score)
            {
                continue;
            }

            Candidate candidate;
            candidate.frame_index = i;
            candidate.center = center;
            candidate.score = score;
            local_candidates.push_back(candidate);
        }

        if (local_candidates.empty())
        {
            candidates_by_target[target_index].push_back(Candidate{});
            continue;
        }

        std::sort(local_candidates.begin(), local_candidates.end(), [](const Candidate& a, const Candidate& b) {
            if (a.score != b.score) return a.score > b.score;
            return a.center < b.center;
        });
        if (local_candidates.size() > 6)
        {
            local_candidates.resize(6);
        }
        std::sort(local_candidates.begin(), local_candidates.end(), [](const Candidate& a, const Candidate& b) {
            return a.center < b.center;
        });
        local_candidates.push_back(Candidate{});
        candidates_by_target[target_index] = std::move(local_candidates);
    }

    const double skip_penalty = 0.12;
    std::vector<std::vector<double>> dp(targets.size());
    std::vector<std::vector<int>> back(targets.size());
    for (size_t target_index = 0; target_index < targets.size(); ++target_index)
    {
        const auto& candidates = candidates_by_target[target_index];
        dp[target_index].assign(candidates.size(), -1.0e18);
        back[target_index].assign(candidates.size(), -1);
        for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index)
        {
            const Candidate& candidate = candidates[candidate_index];
            const double emit_score = (candidate.frame_index >= 0) ? candidate.score : -skip_penalty;
            if (target_index == 0)
            {
                dp[target_index][candidate_index] = emit_score;
                continue;
            }

            const auto& prev_candidates = candidates_by_target[target_index - 1];
            double best_total = -1.0e18;
            int best_prev = -1;
            for (size_t prev_index = 0; prev_index < prev_candidates.size(); ++prev_index)
            {
                const Candidate& prev_candidate = prev_candidates[prev_index];
                if (prev_candidate.frame_index >= 0 && candidate.frame_index >= 0 &&
                    prev_candidate.frame_index >= candidate.frame_index)
                {
                    continue;
                }

                double transition_penalty = 0.0;
                if (prev_candidate.frame_index >= 0 && candidate.frame_index >= 0)
                {
                    const double prior_gap = std::max(0.020, targets[target_index].prior_center - targets[target_index - 1].prior_center);
                    const double actual_gap = candidate.center - prev_candidate.center;
                    const double gap_norm = std::max(0.060, prior_gap * 1.5);
                    transition_penalty += 0.30 * std::min(2.0, std::abs(actual_gap - prior_gap) / gap_norm);
                }
                else if ((prev_candidate.frame_index >= 0) != (candidate.frame_index >= 0))
                {
                    transition_penalty += 0.04;
                }

                const double total = dp[target_index - 1][prev_index] + emit_score - transition_penalty;
                if (total > best_total)
                {
                    best_total = total;
                    best_prev = static_cast<int>(prev_index);
                }
            }
            dp[target_index][candidate_index] = best_total;
            back[target_index][candidate_index] = best_prev;
        }
    }

    int best_final_index = -1;
    double best_final_score = -1.0e18;
    for (size_t candidate_index = 0; candidate_index < candidates_by_target.back().size(); ++candidate_index)
    {
        if (dp.back()[candidate_index] > best_final_score)
        {
            best_final_score = dp.back()[candidate_index];
            best_final_index = static_cast<int>(candidate_index);
        }
    }

    std::vector<int> chosen_candidate_by_target(targets.size(), -1);
    int current_index = best_final_index;
    for (int target_index = static_cast<int>(targets.size()) - 1; target_index >= 0; --target_index)
    {
        chosen_candidate_by_target[static_cast<size_t>(target_index)] = current_index;
        current_index = (current_index >= 0) ? back[static_cast<size_t>(target_index)][static_cast<size_t>(current_index)] : -1;
    }

    for (size_t target_index = 0; target_index < targets.size(); ++target_index)
    {
        const int chosen_index = chosen_candidate_by_target[target_index];
        if (chosen_index < 0)
        {
            continue;
        }
        const Candidate& chosen = candidates_by_target[target_index][static_cast<size_t>(chosen_index)];
        if (chosen.frame_index < 0)
        {
            continue;
        }

        const auto& frame = frames[chosen.frame_index];
        LandmarkObservation obs;
        obs.target_index = targets[target_index].target_index;
        obs.type = targets[target_index].type;
        obs.frame_index = chosen.frame_index;
        obs.start = frame.AudioBufferStartSec;
        obs.end = frame.AudioBufferEndSec;
        obs.center = frame.AudioBufferCenterSec;
        obs.score = chosen.score;
        obs.confidence = clamp01(
            chosen.score * 0.60 +
            std::max(0.0, chosen.score - landmark_conditioned_detection_threshold(targets[target_index].type)) * 0.60);
        obs.decision_sec = frame.AudioBufferEndSec;
        obs.top_class = targets[target_index].type;
        obs.source = "conditioned_phone";
        observations.push_back(std::move(obs));
    }

    std::vector<LandmarkObservation> pause_observations = detect_pause_boundary_observations();
    observations.insert(observations.end(), pause_observations.begin(), pause_observations.end());

    std::sort(observations.begin(), observations.end(), [](const LandmarkObservation& a, const LandmarkObservation& b) {
        if (a.center != b.center) return a.center < b.center;
        return a.type < b.type;
    });
    return observations;
}

static std::string audio_landmark_observations_csv(const std::vector<LandmarkObservation>& observations)
{
    std::ostringstream out;
    out << "observation_index,target_index,type,frame_index,start,end,center,score,confidence,decision_sec,top_class,source,matched_target_index,matched_error_ms\n";
    out << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < observations.size(); ++i)
    {
        const auto& obs = observations[i];
        out << i << ','
            << obs.target_index << ','
            << obs.type << ','
            << obs.frame_index << ','
            << obs.start << ','
            << obs.end << ','
            << obs.center << ','
            << obs.score << ','
            << obs.confidence << ','
            << obs.decision_sec << ','
            << obs.top_class << ','
            << obs.source << ','
            << obs.matched_target_index << ','
            << obs.matched_error_ms << '\n';
    }
    return out.str();
}

static LandmarkAuditReport grade_landmark_audit(
    std::vector<LandmarkObservation>& observations,
    const std::vector<LandmarkTarget>& targets,
    const TArray<FOffgridAIStreamingAudioFeatureFrame>& frames)
{
    LandmarkAuditReport report;
    report.available = !targets.empty() && frames.Num() > 0;
    report.target_count = static_cast<int>(targets.size());
    report.observation_count = static_cast<int>(observations.size());
    if (!report.available)
    {
        return report;
    }

    const std::vector<std::string> types = {"mbp", "fv", "w", "chjjsh", "round", "pause_lull"};
    const std::vector<std::pair<std::string, double>> score_thresholds = {
        {"0.60", 0.60},
        {"0.70", 0.70},
        {"0.80", 0.80},
        {"0.90", 0.90},
    };
    std::map<std::string, std::vector<double>> target_window_scores;
    std::map<std::string, int> target_window_hits;

    for (const auto& type : types)
    {
        report.by_type[type] = LandmarkTypeReport{};
        for (const auto& [label, _] : score_thresholds)
        {
            report.by_type[type].threshold_observation_count[label] = 0;
            report.by_type[type].threshold_matched_observation_count[label] = 0;
            report.by_type[type].threshold_matched_target_count[label] = 0;
        }
    }
    report.boundary_events["pause_close"] = BoundaryEventAuditReport{};
    report.boundary_events["resume"] = BoundaryEventAuditReport{};

    for (const LandmarkTarget& target : targets)
    {
        if (!target.has_gold_reference)
        {
            continue;
        }
        auto& stats = report.by_type[target.type];
        ++stats.target_count;

        double best_score = 0.0;
        const double tolerance = landmark_match_tolerance_seconds(target.type);
        for (int32 i = 0; i < frames.Num(); ++i)
        {
            const auto& frame = frames[i];
            if (std::abs(static_cast<double>(frame.AudioBufferCenterSec) - target.gold_center) > tolerance)
            {
                continue;
            }
            const double score = is_pause_lull_type(target.type)
                ? comma_lull_score_for_frame(frame)
                : landmark_score_for_type(
                    target.type,
                    FOffgridAIOnlinePhoneAligner::BuildArticulatoryProbabilityField(frame));
            best_score = std::max(best_score, score);
        }
        target_window_scores[target.type].push_back(best_score);
        if (best_score >= landmark_detection_threshold(target.type))
        {
            ++target_window_hits[target.type];
        }
    }

    std::map<std::string, std::vector<int>> observations_by_type;
    for (size_t i = 0; i < observations.size(); ++i)
    {
        ++report.by_type[observations[i].type].observation_count;
        for (const auto& [label, threshold] : score_thresholds)
        {
            if (observations[i].score >= threshold)
            {
                ++report.by_type[observations[i].type].threshold_observation_count[label];
            }
        }
        observations_by_type[observations[i].type].push_back(static_cast<int>(i));
    }

    std::map<int, int> observation_index_by_target;
    for (size_t i = 0; i < observations.size(); ++i)
    {
        if (observations[i].target_index >= 0)
        {
            observation_index_by_target[observations[i].target_index] = static_cast<int>(i);
        }
    }

    auto match_boundary_events = [](
        const std::vector<const LandmarkObservation*>& event_observations,
        const std::vector<double>& target_times,
        double tolerance_ms,
        bool bUseStartTime) -> std::vector<MonotonicBoundaryMatch>
    {
        std::vector<MonotonicBoundaryMatch> matches;
        size_t obs_index = 0;
        size_t target_index = 0;
        while (obs_index < event_observations.size() && target_index < target_times.size())
        {
            const LandmarkObservation* observation = event_observations[obs_index];
            const double observation_time = bUseStartTime ? observation->start : observation->end;
            const double target_time = target_times[target_index];
            const double error_ms = std::abs(observation_time - target_time) * 1000.0;

            if (error_ms <= tolerance_ms)
            {
                MonotonicBoundaryMatch match;
                match.observation_index = static_cast<int>(obs_index);
                match.target_index = static_cast<int>(target_index);
                match.error_ms = error_ms;
                match.latency_ms = std::max(0.0, (observation->decision_sec - observation_time) * 1000.0);
                matches.push_back(match);
                ++obs_index;
                ++target_index;
                continue;
            }

            if (observation_time < target_time)
            {
                ++obs_index;
            }
            else
            {
                ++target_index;
            }
        }
        return matches;
    };

    for (const std::string& type : types)
    {
        auto& stats = report.by_type[type];
        std::vector<double> match_errors_ms;
        std::vector<double> matched_scores;

        if (type == "pause_lull")
        {
            std::vector<const LandmarkObservation*> pause_observations;
            std::vector<double> pause_target_centers;
            for (const LandmarkObservation& observation : observations)
            {
                if (observation.type == "pause_lull" && observation.source == "boundary_interval")
                {
                    pause_observations.push_back(&observation);
                }
            }
            for (const LandmarkTarget& target : targets)
            {
                if (target.type == "pause_lull" && target.has_gold_reference)
                {
                    pause_target_centers.push_back(target.gold_center);
                }
            }

            const double tolerance_ms = landmark_match_tolerance_seconds(type) * 1000.0;
            const std::vector<MonotonicBoundaryMatch> matches =
                match_boundary_events(pause_observations, pause_target_centers, tolerance_ms, false);
            for (const MonotonicBoundaryMatch& match : matches)
            {
                ++stats.matched_target_count;
                ++stats.matched_observation_count;
                const LandmarkObservation* observation = pause_observations[static_cast<size_t>(match.observation_index)];
                match_errors_ms.push_back(match.error_ms);
                matched_scores.push_back(observation->score);
                for (const auto& [label, threshold] : score_thresholds)
                {
                    if (observation->score >= threshold)
                    {
                        ++stats.threshold_matched_target_count[label];
                        ++stats.threshold_matched_observation_count[label];
                    }
                }
            }
        }
        else
        {
            for (size_t target_index = 0; target_index < targets.size(); ++target_index)
            {
                const LandmarkTarget& target = targets[target_index];
                if (target.type != type || !target.has_gold_reference)
                {
                    continue;
                }

                const auto obs_it = observation_index_by_target.find(static_cast<int>(target_index));
                if (obs_it == observation_index_by_target.end())
                {
                    continue;
                }
                const int obs_index = obs_it->second;
                LandmarkObservation& observation = observations[static_cast<size_t>(obs_index)];
                const double error_ms = std::abs(observation.center - target.gold_center) * 1000.0;
                const double tolerance_ms = landmark_match_tolerance_seconds(type) * 1000.0;
                if (error_ms <= tolerance_ms)
                {
                    ++stats.matched_target_count;
                    ++stats.matched_observation_count;
                    observation.matched_target_index = static_cast<int>(target_index);
                    observation.matched_error_ms = error_ms;
                    match_errors_ms.push_back(error_ms);
                    matched_scores.push_back(observation.score);
                    for (const auto& [label, threshold] : score_thresholds)
                    {
                        if (observation.score >= threshold)
                        {
                            ++stats.threshold_matched_target_count[label];
                            ++stats.threshold_matched_observation_count[label];
                        }
                    }
                }
            }
        }

        stats.recall = stats.target_count > 0
            ? static_cast<double>(stats.matched_target_count) / static_cast<double>(stats.target_count)
            : 0.0;
        stats.precision = stats.observation_count > 0
            ? static_cast<double>(stats.matched_observation_count) / static_cast<double>(stats.observation_count)
            : 0.0;
        stats.mean_abs_center_error_ms = mean_ms(match_errors_ms);
        stats.median_abs_center_error_ms = percentile_ms(match_errors_ms, 0.50);
        stats.p90_abs_center_error_ms = percentile_ms(match_errors_ms, 0.90);
        stats.mean_target_window_peak_score = mean_ms(target_window_scores[type]);
        stats.mean_matched_observation_score = mean_ms(matched_scores);
        stats.target_window_hit_rate = stats.target_count > 0
            ? static_cast<double>(target_window_hits[type]) / static_cast<double>(stats.target_count)
            : 0.0;

        report.matched_target_count += stats.matched_target_count;
        report.matched_observation_count += stats.matched_observation_count;
    }

    {
        auto& close_stats = report.boundary_events["pause_close"];
        auto& resume_stats = report.boundary_events["resume"];
        std::vector<double> close_errors_ms;
        std::vector<double> close_latencies_ms;
        std::vector<double> resume_errors_ms;
        std::vector<double> resume_latencies_ms;

        std::vector<const LandmarkTarget*> pause_targets;
        pause_targets.reserve(targets.size());
        for (const LandmarkTarget& target : targets)
        {
            if (target.type == "pause_lull" && target.has_gold_reference)
            {
                pause_targets.push_back(&target);
            }
        }

        std::vector<const LandmarkObservation*> close_observations;
        std::vector<const LandmarkObservation*> resume_observations;
        close_observations.reserve(observations.size());
        resume_observations.reserve(observations.size());
        for (const LandmarkObservation& observation : observations)
        {
            if (observation.type == "pause_close" && observation.source == "boundary_close")
            {
                close_observations.push_back(&observation);
            }
            if (observation.type == "resume" && observation.source == "boundary_resume")
            {
                resume_observations.push_back(&observation);
            }
        }

        close_stats.target_count = static_cast<int>(pause_targets.size());
        resume_stats.target_count = static_cast<int>(pause_targets.size());
        close_stats.observation_count = static_cast<int>(close_observations.size());
        resume_stats.observation_count = static_cast<int>(resume_observations.size());

        std::vector<double> close_target_times;
        std::vector<double> resume_target_times;
        close_target_times.reserve(pause_targets.size());
        resume_target_times.reserve(pause_targets.size());
        for (const LandmarkTarget* target : pause_targets)
        {
            close_target_times.push_back(target->gold_start);
            resume_target_times.push_back(target->gold_end);
        }
        const double tolerance_ms = landmark_match_tolerance_seconds("pause_lull") * 1000.0;
        const std::vector<MonotonicBoundaryMatch> close_matches =
            match_boundary_events(close_observations, close_target_times, tolerance_ms, true);
        const std::vector<MonotonicBoundaryMatch> resume_matches =
            match_boundary_events(resume_observations, resume_target_times, tolerance_ms, false);

        for (const MonotonicBoundaryMatch& match : close_matches)
        {
            ++close_stats.matched_target_count;
            ++close_stats.matched_observation_count;
            close_errors_ms.push_back(match.error_ms);
            close_latencies_ms.push_back(match.latency_ms);
        }
        for (const MonotonicBoundaryMatch& match : resume_matches)
        {
            ++resume_stats.matched_target_count;
            ++resume_stats.matched_observation_count;
            resume_errors_ms.push_back(match.error_ms);
            resume_latencies_ms.push_back(match.latency_ms);
        }

        close_stats.precision = close_stats.observation_count > 0
            ? static_cast<double>(close_stats.matched_observation_count) / static_cast<double>(close_stats.observation_count)
            : 0.0;
        close_stats.recall = close_stats.target_count > 0
            ? static_cast<double>(close_stats.matched_target_count) / static_cast<double>(close_stats.target_count)
            : 0.0;
        close_stats.mean_abs_error_ms = mean_ms(close_errors_ms);
        close_stats.median_abs_error_ms = percentile_ms(close_errors_ms, 0.50);
        close_stats.p90_abs_error_ms = percentile_ms(close_errors_ms, 0.90);
        close_stats.mean_decision_latency_ms = mean_ms(close_latencies_ms);
        close_stats.median_decision_latency_ms = percentile_ms(close_latencies_ms, 0.50);
        close_stats.p90_decision_latency_ms = percentile_ms(close_latencies_ms, 0.90);

        resume_stats.precision = resume_stats.observation_count > 0
            ? static_cast<double>(resume_stats.matched_observation_count) / static_cast<double>(resume_stats.observation_count)
            : 0.0;
        resume_stats.recall = resume_stats.target_count > 0
            ? static_cast<double>(resume_stats.matched_target_count) / static_cast<double>(resume_stats.target_count)
            : 0.0;
        resume_stats.mean_abs_error_ms = mean_ms(resume_errors_ms);
        resume_stats.median_abs_error_ms = percentile_ms(resume_errors_ms, 0.50);
        resume_stats.p90_abs_error_ms = percentile_ms(resume_errors_ms, 0.90);
        resume_stats.mean_decision_latency_ms = mean_ms(resume_latencies_ms);
        resume_stats.median_decision_latency_ms = percentile_ms(resume_latencies_ms, 0.50);
        resume_stats.p90_decision_latency_ms = percentile_ms(resume_latencies_ms, 0.90);
    }

    report.target_count = 0;
    report.observation_count = 0;
    for (const auto& [_, stats] : report.by_type)
    {
        report.target_count += stats.target_count;
        report.observation_count += stats.observation_count;
    }

    return report;
}

static std::string landmark_audit_json(const LandmarkAuditReport& report)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(6);
    out << "{\n"
        << "  \"available\": " << (report.available ? "true" : "false") << ",\n"
        << "  \"target_count\": " << report.target_count << ",\n"
        << "  \"observation_count\": " << report.observation_count << ",\n"
        << "  \"matched_target_count\": " << report.matched_target_count << ",\n"
        << "  \"matched_observation_count\": " << report.matched_observation_count << ",\n"
        << "  \"types\": {\n";

    bool first = true;
    for (const auto& [type, stats] : report.by_type)
    {
        if (!first)
        {
            out << ",\n";
        }
        first = false;
        out << "    \"" << type << "\": {\n"
            << "      \"target_count\": " << stats.target_count << ",\n"
            << "      \"observation_count\": " << stats.observation_count << ",\n"
            << "      \"matched_target_count\": " << stats.matched_target_count << ",\n"
            << "      \"matched_observation_count\": " << stats.matched_observation_count << ",\n"
            << "      \"recall\": " << stats.recall << ",\n"
            << "      \"precision\": " << stats.precision << ",\n"
            << "      \"mean_abs_center_error_ms\": " << stats.mean_abs_center_error_ms << ",\n"
            << "      \"median_abs_center_error_ms\": " << stats.median_abs_center_error_ms << ",\n"
            << "      \"p90_abs_center_error_ms\": " << stats.p90_abs_center_error_ms << ",\n"
            << "      \"mean_target_window_peak_score\": " << stats.mean_target_window_peak_score << ",\n"
            << "      \"mean_matched_observation_score\": " << stats.mean_matched_observation_score << ",\n"
            << "      \"target_window_hit_rate\": " << stats.target_window_hit_rate << ",\n"
            << "      \"thresholds\": {\n";
        bool first_threshold = true;
        for (const auto& [label, observation_count] : stats.threshold_observation_count)
        {
            if (!first_threshold)
            {
                out << ",\n";
            }
            first_threshold = false;
            const auto matched_observation_it = stats.threshold_matched_observation_count.find(label);
            const auto matched_target_it = stats.threshold_matched_target_count.find(label);
            const int matched_observation_count = matched_observation_it != stats.threshold_matched_observation_count.end()
                ? matched_observation_it->second
                : 0;
            const int matched_target_count = matched_target_it != stats.threshold_matched_target_count.end()
                ? matched_target_it->second
                : 0;
            out << "        \"" << label << "\": {\n"
                << "          \"observation_count\": " << observation_count << ",\n"
                << "          \"matched_observation_count\": " << matched_observation_count << ",\n"
                << "          \"matched_target_count\": " << matched_target_count << ",\n"
                << "          \"precision\": "
                << (observation_count > 0 ? static_cast<double>(matched_observation_count) / static_cast<double>(observation_count) : 0.0) << ",\n"
                << "          \"recall\": "
                << (stats.target_count > 0 ? static_cast<double>(matched_target_count) / static_cast<double>(stats.target_count) : 0.0) << "\n"
                << "        }";
        }
        out << "\n"
            << "      }\n"
            << "    }";
    }
    out << "\n  },\n"
        << "  \"boundary_events\": {\n";

    bool first_boundary = true;
    for (const auto& [name, stats] : report.boundary_events)
    {
        if (!first_boundary)
        {
            out << ",\n";
        }
        first_boundary = false;
        out << "    \"" << name << "\": {\n"
            << "      \"target_count\": " << stats.target_count << ",\n"
            << "      \"observation_count\": " << stats.observation_count << ",\n"
            << "      \"matched_target_count\": " << stats.matched_target_count << ",\n"
            << "      \"matched_observation_count\": " << stats.matched_observation_count << ",\n"
            << "      \"precision\": " << stats.precision << ",\n"
            << "      \"recall\": " << stats.recall << ",\n"
            << "      \"mean_abs_error_ms\": " << stats.mean_abs_error_ms << ",\n"
            << "      \"median_abs_error_ms\": " << stats.median_abs_error_ms << ",\n"
            << "      \"p90_abs_error_ms\": " << stats.p90_abs_error_ms << ",\n"
            << "      \"mean_decision_latency_ms\": " << stats.mean_decision_latency_ms << ",\n"
            << "      \"median_decision_latency_ms\": " << stats.median_decision_latency_ms << ",\n"
            << "      \"p90_decision_latency_ms\": " << stats.p90_decision_latency_ms << "\n"
            << "    }";
    }

    out << "\n  }\n"
        << "}\n";
    return out.str();
}

static OraclePhoneDurationReport apply_gold_phone_durations_to_plan(
    FOffgridAITextVisemePlan& plan,
    const std::vector<GoldPhoneTiming>& gold_phones)
{
    OraclePhoneDurationReport report;
    if (plan.ExpectedPhones.Num() <= 0 || gold_phones.empty())
    {
        report.unmatched_count = plan.ExpectedPhones.Num();
        return report;
    }

    std::vector<const GoldPhoneTiming*> by_global_index(gold_phones.size(), nullptr);
    for (const GoldPhoneTiming& phone : gold_phones)
    {
        if (phone.global_phone_index >= 0 && static_cast<size_t>(phone.global_phone_index) < by_global_index.size())
        {
            by_global_index[static_cast<size_t>(phone.global_phone_index)] = &phone;
        }
    }

    double duration_sum = 0.0;
    for (int32 phone_index = 0; phone_index < plan.ExpectedPhones.Num(); ++phone_index)
    {
        FOffgridAIExpectedPhone& expected = plan.ExpectedPhones[phone_index];
        const GoldPhoneTiming* matched_phone = nullptr;

        if (phone_index >= 0 && static_cast<size_t>(phone_index) < by_global_index.size())
        {
            matched_phone = by_global_index[static_cast<size_t>(phone_index)];
            if (matched_phone
                && (matched_phone->phone != to_std(expected.Phone) || matched_phone->word_index != expected.WordIndex))
            {
                matched_phone = nullptr;
            }
        }

        if (matched_phone)
        {
            ++report.exact_global_matches;
        }
        else
        {
            for (const GoldPhoneTiming& phone : gold_phones)
            {
                if (phone.word_index != expected.WordIndex) continue;
                if (phone.word_phone_index != expected.WordPhoneIndex) continue;
                if (phone.phone != to_std(expected.Phone)) continue;
                matched_phone = &phone;
                ++report.fallback_word_matches;
                break;
            }
        }

        if (!matched_phone)
        {
            ++report.unmatched_count;
            continue;
        }

        const float observed_duration = static_cast<float>(std::max(0.0, matched_phone->end - matched_phone->start));
        expected.WeightSeconds = observed_duration;
        duration_sum += observed_duration;
        ++report.matched_count;
    }

    if (report.matched_count > 0)
    {
        report.mean_duration_seconds = duration_sum / static_cast<double>(report.matched_count);
        plan.EstimatedDurationSeconds = static_cast<float>(std::max(
            static_cast<double>(plan.EstimatedDurationSeconds),
            duration_sum));
    }
    return report;
}

static std::string gold_visible_visemes_csv(const std::vector<HandmadeLabel>& labels)
{
    std::ostringstream out;
    out << "start,end,pose,word,confidence,word_index,speech_region_index,text_sentence_index\n";
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
            << label.sentence_index << '\n';
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

    std::vector<const GoldPhoneTiming*> by_global_index(gold_phones.size(), nullptr);
    for (const GoldPhoneTiming& phone : gold_phones)
    {
        if (phone.global_phone_index >= 0 && static_cast<size_t>(phone.global_phone_index) < by_global_index.size())
        {
            by_global_index[static_cast<size_t>(phone.global_phone_index)] = &phone;
        }
    }

    for (const FOffgridAITextVisemeEvent& event : plan.Events)
    {
        const GoldPhoneTiming* matched_phone = nullptr;
        if (event.SourcePhoneGlobalIndex >= 0
            && static_cast<size_t>(event.SourcePhoneGlobalIndex) < by_global_index.size())
        {
            matched_phone = by_global_index[static_cast<size_t>(event.SourcePhoneGlobalIndex)];
        }

        if (!matched_phone)
        {
            for (const GoldPhoneTiming& phone : gold_phones)
            {
                if (phone.word_index != event.WordIndex) continue;
                if (phone.word_phone_index != event.SourcePhoneIndex) continue;
                matched_phone = &phone;
                break;
            }
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
    out << "index,pose,word,word_index,text_sentence_index,text_center_norm,strength,source_phone,source_phone_index,generator\n";
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
            << to_std(event.SourcePhoneBase) << ','
            << event.SourcePhoneIndex << ','
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

static std::string planned_boundaries_csv(const FOffgridAITextVisemePlan& plan, const std::vector<LandmarkTarget>& targets)
{
    std::map<int32, const LandmarkTarget*> pause_targets_by_word_index;
    for (const LandmarkTarget& target : targets)
    {
        if (target.type == "pause_lull" && target.word_index >= 0)
        {
            pause_targets_by_word_index[target.word_index] = &target;
        }
    }

    std::ostringstream out;
    out << "word_index,next_word_index,boundary_mark,pause_class,pause_seconds,prior_start,prior_end,prior_center\n";
    out << std::fixed << std::setprecision(6);
    const int32 boundary_count = std::min(plan.WordBoundaryPunctuationAfter.Num(), plan.WordBoundaryPauseClassAfter.Num());
    for (int32 word_index = 0; word_index < boundary_count; ++word_index)
    {
        const TCHAR boundary_mark = plan.WordBoundaryPunctuationAfter[word_index];
        const EOffgridAIBoundaryPauseClass pause_class = plan.WordBoundaryPauseClassAfter[word_index];
        const auto it = pause_targets_by_word_index.find(word_index);
        const LandmarkTarget* target = (it != pause_targets_by_word_index.end()) ? it->second : nullptr;
        out << word_index << ','
            << (word_index + 1) << ','
            << static_cast<int>(boundary_mark) << ','
            << pause_class_name(pause_class) << ','
            << (plan.WordBoundaryPauseSecondsAfter.IsValidIndex(word_index)
                ? plan.WordBoundaryPauseSecondsAfter[word_index]
                : 0.0f) << ','
            << (target ? target->prior_start : 0.0) << ','
            << (target ? target->prior_end : 0.0) << ','
            << (target ? target->prior_center : 0.0)
            << '\n';
    }
    return out.str();
}

static std::string expected_phones_csv(const FOffgridAITextVisemePlan& plan)
{
    std::ostringstream out;
    out << "phone_index,word_phone_index,phone,base_phone,word,word_index,speech_region_index,text_sentence_index,is_vowel,is_visible_viseme,first_visible_event_index,boundary_after_word,pause_class_after_word,pause_seconds_after_word,weight_seconds\n";
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
            << phone.FirstVisibleEventIndex << ','
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
    out << "index,start,end,center,rms,rms_norm,delta_rms,flux,zcr,low_band,mid_band,high_band,centroid,periodicity,evidence,open_threshold,close_threshold,in_speech_before,in_speech_after,open_candidate,keep_open,strong_onset,strong_quiet,low_evidence,endpoint_active,silence_accum,endpoint_start,active_speech_region_start,active_speech_region_end,frame_started_speech_region,frame_closed_speech_region,frame_bridged_speech_region,decision,local_rms_peak,local_rms_valley,local_flux_peak,derived_pause_family,derived_pause_confidence\n";
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
            << derived_pause_confidence(f) << '\n';
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
            const float Score = FOffgridAIOnlinePhoneAligner::ScoreForClass(Scores, PhoneClass);
            if (ClassIndex == 0 || Score > Best.Score)
            {
                Best.Class = PhoneClass;
                Best.Score = Score;
            }
        }

        OutScore = Best.Score;
        return to_std(FOffgridAIOnlinePhoneAligner::PhoneClassToString(Best.Class));
    };

    for (int32 i = 0; i < frames.Num(); ++i)
    {
        const auto& f = frames[i];
        const FOffgridAIArticulatoryProbabilityField field = FOffgridAIOnlinePhoneAligner::BuildArticulatoryProbabilityField(f);
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
    out << "index,start,center,end,pose,word,word_index,speech_region_index,text_sentence_index,strength,reason,source_phone_index,source_phone_base,source_phone_class,text_center_norm,text_diagnostic_center,prior_start,prior_center,prior_end,lead_adjusted_center,playback_offset,total_paused_at_commit,min_live_lead_delay,inter_event_floor_delay,total_center_delay,commit_playback,commit_lead,mapped_to_observed_speech,used_initial_speech_anchor,used_resume_anchor,acoustic_anchor_kind,acoustic_anchor_seconds,acoustic_anchor_error_seconds,observed_pause_decay_seconds,observed_resume_onset_seconds,observed_resume_energy_anchor_seconds,boundary_word_index,boundary_mark,boundary_outcome\n";
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
    out << "event_index,pose,word,word_index,source_phone_index,source_phone_base,source_phone_class,commit_reason,render_start,render_center,render_end,prior_start,prior_center,prior_end,lead_adjusted_center,playback_offset,total_paused_at_commit,min_live_lead_delay,inter_event_floor_delay,total_center_delay,commit_playback,commit_lead,mapped_to_observed_speech,used_initial_speech_anchor,used_resume_anchor,acoustic_anchor_kind,acoustic_anchor_seconds,acoustic_anchor_error_seconds,observed_pause_decay_seconds,observed_resume_onset_seconds,observed_resume_energy_anchor_seconds,boundary_word_index,boundary_mark,boundary_outcome\n";
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
    out << "line_id,update_ordinal,final_replay,current_playback_sec,b_playhead_started,b_hold_active,b_observed_pause_lull,b_saw_confirmed_out_of_speech_after_boundary,b_resume_anchor_active,b_resume_anchor_from_initial_speech,boundary_word_index,boundary_mark,pause_class,hold_start_speech_region_index,resume_anchor_event_index,playback_origin_sec,last_playback_sec,active_playhead_sec,total_paused_sec,hold_start_playback_sec,boundary_search_start_playback_sec,hold_deadline_playback_sec,hold_resume_target_active_sec,hold_resume_target_lead_sec,confirmed_quiet_start_playback_sec,quiet_rms_norm_at_decay,quiet_evidence_at_decay,quiet_raw_rms_at_decay,observed_resume_onset_playback_sec,observed_resume_energy_anchor_sec,resume_anchor_active_sec,resume_anchor_lead_sec,resume_anchor_final_center_sec,b_syllable_rebase_active,syllable_anchor_phone_index,next_expected_syllable_phone_index,syllable_anchor_active_sec,syllable_anchor_audio_sec,syllable_rate,syllable_section_start_audio_sec,syllable_assignment_confidence,syllable_assignment_margin,syllable_assignment_skip_count,syllable_observed_audio_sec,syllable_pulse_count,syllable_assignment_count,syllable_last_assigned_phone_index,syllable_reject_low_prominence_count,syllable_reject_before_section_count,syllable_late_assignment_count,syllable_reject_no_candidate_count,syllable_ambiguous_assignment_count,syllable_duplicate_pulse_count,b_strong_phone_rebase_active,strong_phone_anchor_phone_index,strong_phone_anchor_active_sec,strong_phone_anchor_audio_sec,strong_phone_anchor_confidence,strong_phone_candidate_count,strong_phone_assignment_count,transcript_anchor_cursor_phone_index,transcript_anchor_fence_word_index,transcript_anchor_fence_phone_index,transcript_anchor_last_resolved_boundary_word_index,last_fence_estimator_outcome,last_resolved_boundary_outcome,last_resolved_boundary_word_index,last_resolved_boundary_mark,last_resolved_boundary_decay_sec,last_resolved_boundary_resume_onset_sec,last_resolved_boundary_resume_energy_anchor_sec,committed_event_count,diagnostic_kind\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& row : rows)
    {
        out << to_std(row.LineID) << ','
            << row.UpdateOrdinal << ','
            << (row.bFinalReplay ? 1 : 0) << ','
            << row.CurrentPlaybackSec << ','
            << (row.bPlayheadStarted ? 1 : 0) << ','
            << (row.bHoldActive ? 1 : 0) << ','
            << (row.bObservedPauseLull ? 1 : 0) << ','
            << (row.bSawConfirmedOutOfSpeechAfterBoundary ? 1 : 0) << ','
            << (row.bResumeAnchorActive ? 1 : 0) << ','
            << (row.bResumeAnchorFromInitialSpeech ? 1 : 0) << ','
            << row.BoundaryWordIndex << ','
            << csv_value(to_std(row.BoundaryMark)) << ','
            << to_std(row.PauseClass) << ','
            << row.HoldStartSpeechRegionIndex << ','
            << row.ResumeAnchorEventIndex << ','
            << row.PlaybackOriginSec << ','
            << row.LastPlaybackSec << ','
            << row.ActivePlayheadSec << ','
            << row.TotalPausedSec << ','
            << row.HoldStartPlaybackSec << ','
            << row.BoundarySearchStartPlaybackSec << ','
            << row.HoldDeadlinePlaybackSec << ','
            << row.HoldResumeTargetActiveSec << ','
            << row.HoldResumeTargetLeadSec << ','
            << row.ConfirmedQuietStartPlaybackSec << ','
            << row.QuietRMSNormAtDecay << ','
            << row.QuietEvidenceAtDecay << ','
            << row.QuietRawRMSAtDecay << ','
            << row.ObservedResumeOnsetPlaybackSec << ','
            << row.ObservedResumeEnergyAnchorSec << ','
            << row.ResumeAnchorActiveSec << ','
            << row.ResumeAnchorLeadSec << ','
            << row.ResumeAnchorFinalCenterSec << ','
            << (row.bSyllableRebaseActive ? 1 : 0) << ','
            << row.SyllableAnchorPhoneIndex << ','
            << row.NextExpectedSyllablePhoneIndex << ','
            << row.SyllableAnchorActiveSec << ','
            << row.SyllableAnchorAudioSec << ','
            << row.SyllableRate << ','
            << row.SyllableSectionStartAudioSec << ','
            << row.SyllableAssignmentConfidence << ','
            << row.SyllableAssignmentMargin << ','
            << row.SyllableAssignmentSkipCount << ','
            << row.SyllableObservedAudioSec << ','
            << row.SyllablePulseCount << ','
            << row.SyllableAssignmentCount << ','
            << row.SyllableLastAssignedPhoneIndex << ','
            << row.SyllableRejectLowProminenceCount << ','
            << row.SyllableRejectBeforeSectionCount << ','
            << row.SyllableLateAssignmentCount << ','
            << row.SyllableRejectNoCandidateCount << ','
            << row.SyllableAmbiguousAssignmentCount << ','
            << row.SyllableDuplicatePulseCount << ','
            << (row.bStrongPhoneRebaseActive ? 1 : 0) << ','
            << row.StrongPhoneAnchorPhoneIndex << ','
            << row.StrongPhoneAnchorActiveSec << ','
            << row.StrongPhoneAnchorAudioSec << ','
            << row.StrongPhoneAnchorConfidence << ','
            << row.StrongPhoneCandidateCount << ','
            << row.StrongPhoneAssignmentCount << ','
            << row.TranscriptAnchorCursorPhoneIndex << ','
            << row.TranscriptAnchorFenceWordIndex << ','
            << row.TranscriptAnchorFencePhoneIndex << ','
            << row.TranscriptAnchorLastResolvedBoundaryWordIndex << ','
            << to_std(row.LastFenceEstimatorOutcome) << ','
            << to_std(row.LastResolvedBoundaryOutcome) << ','
            << row.LastResolvedBoundaryWordIndex << ','
            << csv_value(to_std(row.LastResolvedBoundaryMark)) << ','
            << row.LastResolvedBoundaryDecaySec << ','
            << row.LastResolvedBoundaryResumeOnsetSec << ','
            << row.LastResolvedBoundaryResumeEnergyAnchorSec << ','
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
    const std::vector<int>& event_indices)
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
                if (to_std(event.PoseID) == label.pose)
                {
                    const double center = (label.start + label.end) * 0.5;
                    const double err_ms = std::abs(static_cast<double>(event.FinalRenderCenterSeconds) - center) * 1000.0;
                    if (err_ms <= 180.0)
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
    std::vector<int> event_indices(track.Events.Num());
    std::iota(event_indices.begin(), event_indices.end(), 0);
    return compute_monotonic_matches_subset(track, handmade, label_indices, event_indices);
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
    report.matched_count = std::min(report.reference_count, report.predicted_count);
    report.missing_count = std::max(0, report.reference_count - report.matched_count);
    report.extra_count = std::max(0, report.predicted_count - report.matched_count);
    report.count_mismatch = report.reference_count != report.predicted_count;

    if (report.matched_count <= 0)
    {
        return report;
    }

    std::vector<double> start_errors;
    std::vector<double> end_errors;
    double lead_sum = 0.0;
    double tail_sum = 0.0;
    for (int i = 0; i < report.matched_count; ++i)
    {
        const auto& predicted_span = predicted[static_cast<size_t>(i)];
        const auto& gold_span = gold[static_cast<size_t>(i)];
        start_errors.push_back(std::abs(predicted_span.start - gold_span.start) * 1000.0);
        end_errors.push_back(std::abs(predicted_span.end - gold_span.end) * 1000.0);
        lead_sum += std::max(0.0, gold_span.start - predicted_span.start) * 1000.0;
        tail_sum += std::max(0.0, predicted_span.end - gold_span.end) * 1000.0;
    }

    const double denom = static_cast<double>(report.matched_count);
    report.mean_abs_start_error_ms = mean_ms(start_errors);
    report.median_abs_start_error_ms = percentile_ms(start_errors, 0.50);
    report.p90_abs_start_error_ms = percentile_ms(start_errors, 0.90);
    report.mean_abs_end_error_ms = mean_ms(end_errors);
    report.median_abs_end_error_ms = percentile_ms(end_errors, 0.50);
    report.p90_abs_end_error_ms = percentile_ms(end_errors, 0.90);
    report.mean_lead_in_ms = lead_sum / denom;
    report.mean_tail_out_ms = tail_sum / denom;
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
            return event.WordIndex == word_index;
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
            if (track.Events[i].WordIndex == word_index)
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
    return report;
}

static SpeechRegionContainmentReport grade_speech_region_containment(
    const FOffgridAIAlignedVisemeTrack& track,
    const TArray<FOffgridAIStreamingSpeechRegion>& speech_regions)
{
    SpeechRegionContainmentReport report;
    report.predicted_region_count = speech_regions.Num();
    report.event_count = track.Events.Num();

    std::vector<double> early_leaks_ms;
    std::vector<double> late_leaks_ms;
    for (const auto& event : track.Events)
    {
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
    report.committed_count = track.Events.Num();
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
            return event.WordIndex == label.word_index;
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

static std::string grade_json(const GradeReport& grade_report)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(3);
    out << "{\n"
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
            if (!cli.case_filter.empty()) std::cerr << "[case] begin " << stem << std::endl;
            session.BeginLine(begin);
            if (!cli.case_filter.empty()) std::cerr << "[case] stream " << stem << std::endl;

            run_streamed_runtime_session(session, wav, stream);
            if (!cli.case_filter.empty()) std::cerr << "[case] export " << stem << std::endl;

            const auto& plan = session.GetTextPlan();
            const auto& speech = session.GetSpeechRegions();
            const auto& gap_candidates = session.GetSpeechDetector().GetGapCandidates();
            const auto& committed = session.GetCommittedTrack();

            const fs::path case_dir = out_root / stem;
            fs::create_directories(case_dir);
            write_text(case_dir / "planned.csv", planned_csv(plan));
            write_text(case_dir / "speech_regions.csv", speech_csv(speech));
            write_text(case_dir / "committed.csv", committed_csv(committed));
            write_text(case_dir / "dropped.csv", dropped_csv(committed));
            write_text(case_dir / "region_drop_diagnostics.csv", region_drop_diagnostics_csv(committed, speech));
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
                if (!cli.case_filter.empty()) std::cerr << "[case] landmark-targets " << stem << std::endl;
                const auto transcript_landmarks = build_landmark_targets(plan, gold_phones, gold_words, gold_boundaries);
                if (!cli.case_filter.empty()) std::cerr << "[case] prosody-targets " << stem << std::endl;
                const auto prosodic_peak_targets = build_prosodic_peak_targets(plan, gold_phones);
                if (!cli.case_filter.empty()) std::cerr << "[case] visible-labels " << stem << std::endl;
                const auto handmade = build_gold_visible_labels(plan, gold_phones);
                gold_viseme_count = handmade.size();
                if (!cli.case_filter.empty()) std::cerr << "[case] grade " << stem << std::endl;
                write_text(case_dir / "planned_words.csv", planned_words_csv(plan));
                write_text(case_dir / "expected_phones.csv", expected_phones_csv(plan));
                write_text(case_dir / "planned_boundaries.csv", planned_boundaries_csv(plan, transcript_landmarks));
                write_text(case_dir / "transcript_landmarks.csv", transcript_landmarks_csv(transcript_landmarks));
                write_text(case_dir / "transcript_prosodic_peaks.csv", prosodic_peak_targets_csv(prosodic_peak_targets));
                write_text(case_dir / "gold_visible_visemes.csv", gold_visible_visemes_csv(handmade));
                report = grade(committed, speech, handmade, gold_words, gold_speech);
                if (output.write_detailed_diagnostics)
                {
                    write_text(case_dir / "audio_landmark_frames.csv", audio_landmark_frames_csv(session.GetSpeechDetector().GetFeatureFrames()));
                }
                std::vector<LandmarkObservation> observations = detect_transcript_conditioned_landmark_observations(
                    session.GetSpeechDetector().GetFeatureFrames(),
                    session.GetSpeechDetector().GetSpeechRegions(),
                    session.GetSpeechDetector().GetGapCandidates(),
                    session.GetSpeechDetector().GetSoftLullCandidates(),
                    transcript_landmarks);
                if (!cli.case_filter.empty()) std::cerr << "[case] landmark-grade " << stem << std::endl;
                LandmarkAuditReport landmark_report = grade_landmark_audit(
                    observations,
                    transcript_landmarks,
                    session.GetSpeechDetector().GetFeatureFrames());
                write_text(case_dir / "streaming_landmark_observations.csv", audio_landmark_observations_csv(observations));
                write_text(case_dir / "streaming_landmark_grade.json", landmark_audit_json(landmark_report));
                write_text(case_dir / "strict_punctuation_alignment_grade.json",
                    strict_punctuation_alignment_json(transcript_landmarks, observations));
                write_text(case_dir / "strict_punctuation_alignment.csv",
                    strict_punctuation_alignment_csv(transcript_landmarks, observations));
                std::vector<ProsodicPeakObservation> raw_prosodic_peaks;
                if (!cli.case_filter.empty()) std::cerr << "[case] prosody " << stem << std::endl;
                std::vector<ProsodicPeakObservation> prosodic_peak_observations = detect_prosodic_peak_observations(
                    plan,
                    prosodic_peak_targets,
                    session.GetSpeechDetector().GetFeatureFrames(),
                    session.GetSpeechDetector().GetSpeechRegions(),
                    raw_prosodic_peaks);
                const ProsodicPeakReport prosodic_peak_report = grade_prosodic_peaks(
                    prosodic_peak_targets,
                    prosodic_peak_observations,
                    raw_prosodic_peaks);
                write_text(case_dir / "streaming_raw_prosodic_peaks.csv", prosodic_peak_observations_csv(raw_prosodic_peaks));
                write_text(case_dir / "streaming_prosodic_peaks.csv", prosodic_peak_observations_csv(prosodic_peak_observations));
                write_text(case_dir / "streaming_prosodic_peak_grade.json", prosodic_peak_grade_json(prosodic_peak_report));
                std::vector<ProsodicPeakObservation> runtime_syllable_observations =
                    runtime_syllable_assignment_observations(
                        prosodic_peak_targets,
                        session.GetRuntimeBoundaryDiagnosticRows());
                const ProsodicPeakReport runtime_syllable_report = grade_prosodic_peaks(
                    prosodic_peak_targets,
                    runtime_syllable_observations,
                    std::vector<ProsodicPeakObservation>());
                write_text(case_dir / "runtime_syllable_assignments.csv",
                    prosodic_peak_observations_csv(runtime_syllable_observations));
                write_text(case_dir / "runtime_syllable_assignment_grade.json",
                    prosodic_peak_grade_json(runtime_syllable_report));
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
