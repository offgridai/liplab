#include "Lipsync/OffgridAILipsyncRuntimeAdapter.h"
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
#include <tuple>
#include <vector>

namespace fs = std::filesystem;

using FOffgridAIAlignedVisemeTrack = FOffgridAICommittedVisemeTrack;
using FOffgridAIAlignedVisemeEvent = FOffgridAICommittedVisemeEvent;

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
    float chunk_seconds = 0.040f;
    float playback_tick_seconds = 0.020f;
};

struct OutputConfig
{
    bool write_detailed_diagnostics = true;
    bool write_monotonic_training_rows = false;
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

static std::vector<fs::path> read_corpus_transcript_paths(const fs::path& root)
{
    const fs::path manifest_path = root / "inputs" / "corpus.csv";
    std::ifstream file(manifest_path);
    if (!file)
    {
        throw std::runtime_error("unified corpus manifest not found: " + manifest_path.string());
    }

    std::string line;
    if (!std::getline(file, line))
    {
        throw std::runtime_error("unified corpus manifest is empty: " + manifest_path.string());
    }
    const std::vector<std::string> header = parse_csv_cells(line);
    const int case_id_col = header_index(header, "case_id");
    const int transcript_col = header_index(header, "transcript");
    if (case_id_col < 0 || transcript_col < 0)
    {
        throw std::runtime_error("unified corpus manifest has an invalid header");
    }

    std::vector<fs::path> paths;
    while (std::getline(file, line))
    {
        if (line.empty()) continue;
        const std::vector<std::string> cells = parse_csv_cells(line);
        const std::string case_id = csv_cell(cells, case_id_col);
        const std::string relative_path = csv_cell(cells, transcript_col);
        if (case_id.empty() || relative_path.empty())
        {
            throw std::runtime_error("unified corpus manifest contains an incomplete row");
        }
        const fs::path transcript_path = root / fs::path(relative_path);
        if (!fs::exists(transcript_path) || transcript_path.stem().string() != case_id)
        {
            throw std::runtime_error("invalid corpus transcript row: " + case_id);
        }
        paths.push_back(transcript_path);
    }
    return paths;
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
    out << "index,pose,word,word_index,text_sentence_index,text_center_norm,strength,source_phone,source_phone_index,visual_role,renderable,generator\n";
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

static std::string committed_csv(const FOffgridAIAlignedVisemeTrack& track)
{
    std::ostringstream out;
    out << "index,start,center,end,pose,word,word_index,speech_region_index,text_sentence_index,strength,renderable,canceled_by_word_handoff,reason,source_phone_index,source_phone_base,source_phone_class,text_center_norm,text_diagnostic_center,prior_start,prior_center,prior_end,lead_adjusted_center,playback_offset,total_paused_at_commit,min_live_lead_delay,inter_event_floor_delay,total_center_delay,commit_playback,commit_lead,mapped_to_observed_speech,used_initial_speech_anchor,used_resume_anchor,acoustic_anchor_kind,acoustic_anchor_seconds,acoustic_anchor_error_seconds,observed_pause_decay_seconds,observed_resume_onset_seconds,observed_resume_energy_anchor_seconds,boundary_word_index,boundary_mark,boundary_outcome\n";
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
    out << "line_id,update_ordinal,final_replay,current_playback_sec,b_playhead_started,b_audio_speech_active,active_speech_region_index,active_region_start_sec,active_region_end_sec,scheduler_next_event_index,scheduler_next_phone_index,scheduler_commit_frontier_sec,scheduler_commit_lead_sec,scheduler_block_reason,committed_event_count,diagnostic_kind\n";
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
            << row.ActiveRegionStartSec << ','
            << row.ActiveRegionEndSec << ','
            << row.SchedulerNextEventIndex << ','
            << row.SchedulerNextPhoneIndex << ','
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
        << frame.RichPeriodicity;
}

static std::string monotonic_audio_frames_csv(
    const TArray<FOffgridAIStreamingAudioFeatureFrame>& frames)
{
    std::ostringstream out;
    out << "schema_version,frame_index,start_sec,end_sec,log_rms,rms_norm,delta_rms,flux,zcr,"
        << "low_band_norm,mid_band_norm,high_band_norm,spectral_centroid_norm,periodicity,"
        << "rich_low_band_norm,rich_mid_band_norm,rich_high_band_norm,rich_spectral_centroid_norm,"
        << "rich_spectral_rolloff_norm,rich_spectral_flatness,rich_spectral_flux,rich_periodicity\n";
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
        bool IsPauseBoundary = false;
        int PunctuationType = 0;
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
        if (word)
        {
            double target_start = targets[index].Center - 0.5 * targets[index].Duration;
            double target_end = targets[index].Center + 0.5 * targets[index].Duration;
            if (first_in_word) target_start = word->start;
            if (last_in_word) target_end = word->end;
            target_end = std::max(target_end, target_start + 0.010);
            targets[index].Center = 0.5 * (target_start + target_end);
            targets[index].Duration = target_end - target_start;
        }
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
            const int32 PreviousWordIndex = targets[index].WordIndex;
            silence.IsPauseBoundary =
                plan.WordBoundaryPauseClassAfter.IsValidIndex(PreviousWordIndex)
                && plan.WordBoundaryPauseClassAfter[PreviousWordIndex]
                    != EOffgridAIBoundaryPauseClass::None;
            if (plan.WordBoundaryPunctuationTypesAfter.IsValidIndex(PreviousWordIndex))
            {
                silence.PunctuationType = static_cast<int>(
                    plan.WordBoundaryPunctuationTypesAfter[PreviousWordIndex]);
            }
            if (silence.IsSentenceBoundary) silence.DurationPrior = 0.120;
            silence.IsRegionEnd = targets[index].IsRegionEnd;
            silence.IsRegionStart = targets[index + 1].IsRegionStart;
            sequence.push_back(std::move(silence));
        }
        const double last_end = targets.back().Center + 0.5 * targets.back().Duration;
        TokenTarget trailing = silence_target(last_end + 0.050, 0.100, 1.0);
        trailing.IsRegionEnd = true;
        if (plan.WordBoundaryPunctuationTypesAfter.IsValidIndex(targets.back().WordIndex))
        {
            trailing.PunctuationType = static_cast<int>(
                plan.WordBoundaryPunctuationTypesAfter[targets.back().WordIndex]);
        }
        sequence.push_back(std::move(trailing));
    }
    targets = std::move(sequence);

    std::ostringstream out;
    out << "schema_version,token_index,event_index,pose,phone_base,word_index,phone_index_in_word,"
        << "phone_global_index,sentence_index,duration_prior_sec,is_vowel,visual_role,text_center_norm,strength,"
        << "is_silence,is_word_start,is_word_end,is_region_start,is_region_end,is_sentence_boundary,is_pause_boundary,"
        << "has_punctuation,punctuation_type,"
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
            << (target.IsPauseBoundary ? 1 : 0) << ','
            << (target.PunctuationType != 0 ? 1 : 0) << ',' << target.PunctuationType << ','
            << target.Center << ',' << target.Duration << ',' << (target.Gold ? 1 : 0) << '\n';
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

static std::string runtime_delivery_grade_json(
    const FOffgridAIAlignedVisemeTrack& track,
    const FOffgridAITextVisemePlan& plan,
    const std::vector<GoldWordTiming>& gold_words,
    const std::vector<GoldSpeechRegion>& gold_speech,
    std::string* out_word_csv)
{
    constexpr double sample_seconds = 0.010;
    constexpr double visible_weight = 0.20;
    constexpr int minimum_visible_samples = 2;

    struct Delivery
    {
        double peak = 0.0;
        int visible_samples = 0;
        double first_visible_sec = std::numeric_limits<double>::infinity();
        double last_visible_sec = -std::numeric_limits<double>::infinity();
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
            row.first_visible_sec = std::min(row.first_visible_sec, seconds);
            row.last_visible_sec = std::max(row.last_visible_sec, seconds);
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
    int severely_compressed_words = 0;
    int stretched_words = 0;
    int measured_word_durations = 0;
    int successful_word_durations = 0;
    std::vector<double> word_duration_ratios;
    std::vector<double> word_duration_abs_errors_ms;
    std::vector<double> word_duration_signed_errors_ms;
    std::vector<double> word_onset_abs_errors_ms;
    std::vector<double> word_onset_signed_errors_ms;
    std::vector<double> word_exit_abs_errors_ms;
    std::vector<double> word_exit_signed_errors_ms;
    std::vector<double> word_interval_coverages;
    std::vector<double> word_interval_precisions;
    std::vector<double> word_interval_ious;
    int zero_overlap_words = 0;
    int low_coverage_words = 0;
    std::ostringstream word_csv;
    word_csv
        << "word_index,word,sentence_index,speech_region_index,gold_start_sec,gold_end_sec,"
        << "gold_duration_ms,planned_events,committed_events,delivered_events,"
        << "runtime_start_sec,runtime_end_sec,runtime_duration_ms,duration_ratio,"
        << "onset_signed_error_ms,onset_abs_error_ms,exit_signed_error_ms,exit_abs_error_ms,"
        << "overlap_ms,spoken_interval_coverage,animation_interval_precision,interval_iou,"
        << "duration_abs_error_ms,duration_signed_error_ms,duration_tolerance_ms,"
        << "duration_within_tolerance,missing,compressed,severely_compressed,stretched,"
        << "first_event_index,last_event_index,"
        << "previous_word_gap_ms,next_word_gap_ms\n";
    for (const auto& gold : gold_words)
    {
        if (planned_by_word[gold.word_index] <= 0) continue;
        ++expected_words;
        const bool missing = delivered_by_word[gold.word_index] <= 0;
        if (missing) ++missing_words;
        const auto found = committed_by_word.find(gold.word_index);
        double runtime_start = std::numeric_limits<double>::infinity();
        double runtime_end = -std::numeric_limits<double>::infinity();
        int first_event_index = -1;
        int last_event_index = -1;
        if (found != committed_by_word.end())
        {
            for (const auto* event : found->second)
            {
                const auto delivered = delivery.find(event->EventIndex);
                if (was_delivered(*event) && delivered != delivery.end())
                {
                    runtime_start = std::min(runtime_start, delivered->second.first_visible_sec);
                    runtime_end = std::max(runtime_end, delivered->second.last_visible_sec);
                }
                if (first_event_index < 0 || event->EventIndex < first_event_index)
                    first_event_index = event->EventIndex;
                if (last_event_index < 0 || event->EventIndex > last_event_index)
                    last_event_index = event->EventIndex;
            }
        }
        const double gold_duration = std::max(0.0, gold.end - gold.start);
        const bool has_runtime = std::isfinite(runtime_start) && std::isfinite(runtime_end);
        const double runtime_duration = has_runtime
            ? std::max(0.0, runtime_end - runtime_start)
            : 0.0;
        const double duration_ratio = gold_duration > 0.0
            ? runtime_duration / gold_duration
            : 0.0;
        const double duration_signed_error_ms = (runtime_duration - gold_duration) * 1000.0;
        const double duration_abs_error_ms = std::abs(duration_signed_error_ms);
        const double onset_signed_error_ms = has_runtime
            ? (runtime_start - gold.start) * 1000.0 : 0.0;
        const double exit_signed_error_ms = has_runtime
            ? (runtime_end - gold.end) * 1000.0 : 0.0;
        const double onset_abs_error_ms = std::abs(onset_signed_error_ms);
        const double exit_abs_error_ms = std::abs(exit_signed_error_ms);
        const double overlap = has_runtime
            ? std::max(0.0, std::min(runtime_end, gold.end)
                - std::max(runtime_start, gold.start))
            : 0.0;
        const double spoken_interval_coverage = gold_duration > 0.0
            ? overlap / gold_duration : 0.0;
        const double animation_interval_precision = runtime_duration > 0.0
            ? overlap / runtime_duration : 0.0;
        const double interval_union = gold_duration + runtime_duration - overlap;
        const double interval_iou = interval_union > 0.0
            ? overlap / interval_union : 0.0;
        const double duration_tolerance_ms = std::max(80.0, gold_duration * 250.0);
        const bool duration_within_tolerance =
            has_runtime && duration_abs_error_ms <= duration_tolerance_ms;
        const int committed_count = found != committed_by_word.end()
            ? static_cast<int>(found->second.size())
            : 0;
        const bool compressed = gold_duration > 0.0
            && runtime_duration + 0.050 < gold_duration
            && duration_ratio < 0.75;
        const bool severely_compressed = gold_duration > 0.0
            && runtime_duration + 0.080 < gold_duration
            && duration_ratio < 0.50;
        const bool stretched = gold_duration > 0.0
            && runtime_duration > gold_duration + 0.050
            && duration_ratio > 1.25;
        if (has_runtime) ++measured_word_durations;
        if (has_runtime)
        {
            word_onset_abs_errors_ms.push_back(onset_abs_error_ms);
            word_onset_signed_errors_ms.push_back(onset_signed_error_ms);
            word_exit_abs_errors_ms.push_back(exit_abs_error_ms);
            word_exit_signed_errors_ms.push_back(exit_signed_error_ms);
        }
        word_interval_coverages.push_back(spoken_interval_coverage);
        word_interval_precisions.push_back(animation_interval_precision);
        word_interval_ious.push_back(interval_iou);
        if (overlap <= 0.0005) ++zero_overlap_words;
        if (spoken_interval_coverage < 0.50) ++low_coverage_words;
        if (duration_within_tolerance) ++successful_word_durations;
        if (gold_duration > 0.0)
        {
            word_duration_ratios.push_back(duration_ratio);
            word_duration_abs_errors_ms.push_back(duration_abs_error_ms);
            word_duration_signed_errors_ms.push_back(duration_signed_error_ms);
        }
        if (compressed) ++compressed_words;
        if (severely_compressed) ++severely_compressed_words;
        if (stretched) ++stretched_words;

        double previous_gap_ms = 0.0;
        double next_gap_ms = 0.0;
        const auto previous = committed_by_word.find(gold.word_index - 1);
        if (has_runtime && previous != committed_by_word.end())
        {
            double previous_end = -std::numeric_limits<double>::infinity();
            for (const auto* event : previous->second)
                previous_end = std::max(previous_end, static_cast<double>(event->RenderEndSeconds));
            if (std::isfinite(previous_end)) previous_gap_ms = (runtime_start - previous_end) * 1000.0;
        }
        const auto next = committed_by_word.find(gold.word_index + 1);
        if (has_runtime && next != committed_by_word.end())
        {
            double next_start = std::numeric_limits<double>::infinity();
            for (const auto* event : next->second)
                next_start = std::min(next_start, static_cast<double>(event->RenderStartSeconds));
            if (std::isfinite(next_start)) next_gap_ms = (next_start - runtime_end) * 1000.0;
        }
        word_csv << gold.word_index << ",\"" << gold.word << "\"," << gold.sentence_index
            << ',' << gold.speech_region_index << ',' << gold.start << ',' << gold.end
            << ',' << gold_duration * 1000.0 << ',' << planned_by_word[gold.word_index]
            << ',' << committed_count << ',' << delivered_by_word[gold.word_index]
            << ',' << (has_runtime ? runtime_start : -1.0)
            << ',' << (has_runtime ? runtime_end : -1.0)
            << ',' << runtime_duration * 1000.0 << ',' << duration_ratio
            << ',' << onset_signed_error_ms << ',' << onset_abs_error_ms
            << ',' << exit_signed_error_ms << ',' << exit_abs_error_ms
            << ',' << overlap * 1000.0 << ',' << spoken_interval_coverage
            << ',' << animation_interval_precision << ',' << interval_iou
            << ',' << duration_abs_error_ms << ',' << duration_signed_error_ms
            << ',' << duration_tolerance_ms << ',' << (duration_within_tolerance ? 1 : 0)
            << ',' << (missing ? 1 : 0) << ',' << (compressed ? 1 : 0)
            << ',' << (severely_compressed ? 1 : 0) << ',' << (stretched ? 1 : 0)
            << ',' << first_event_index << ',' << last_event_index
            << ',' << previous_gap_ms << ',' << next_gap_ms << '\n';
    }
    if (out_word_csv) *out_word_csv = word_csv.str();

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
        << "  \"measured_word_durations\": " << measured_word_durations << ",\n"
        << "  \"successful_word_durations\": " << successful_word_durations << ",\n"
        << "  \"word_duration_success_rate\": " << rate(successful_word_durations, expected_words) << ",\n"
        << "  \"compressed_words\": " << compressed_words << ",\n"
        << "  \"severely_compressed_words\": " << severely_compressed_words << ",\n"
        << "  \"stretched_words\": " << stretched_words << ",\n"
        << "  \"word_duration_abs_error_mean_ms\": " << mean_ms(word_duration_abs_errors_ms) << ",\n"
        << "  \"word_duration_abs_error_median_ms\": " << percentile_ms(word_duration_abs_errors_ms, 0.5) << ",\n"
        << "  \"word_duration_signed_error_mean_ms\": " << mean_ms(word_duration_signed_errors_ms) << ",\n"
        << "  \"word_duration_ratio_mean\": " << mean_ms(word_duration_ratios) << ",\n"
        << "  \"word_duration_ratio_median\": " << percentile_ms(word_duration_ratios, 0.5) << ",\n"
        << "  \"zero_overlap_words\": " << zero_overlap_words << ",\n"
        << "  \"low_coverage_words\": " << low_coverage_words << ",\n"
        << "  \"word_onset_abs_errors_ms\": [";
    for (size_t index = 0; index < word_onset_abs_errors_ms.size(); ++index)
    {
        if (index > 0) out << ',';
        out << word_onset_abs_errors_ms[index];
    }
    out << "],\n"
        << "  \"word_onset_signed_errors_ms\": [";
    for (size_t index = 0; index < word_onset_signed_errors_ms.size(); ++index)
    {
        if (index > 0) out << ',';
        out << word_onset_signed_errors_ms[index];
    }
    out << "],\n"
        << "  \"word_exit_abs_errors_ms\": [";
    for (size_t index = 0; index < word_exit_abs_errors_ms.size(); ++index)
    {
        if (index > 0) out << ',';
        out << word_exit_abs_errors_ms[index];
    }
    out << "],\n"
        << "  \"word_exit_signed_errors_ms\": [";
    for (size_t index = 0; index < word_exit_signed_errors_ms.size(); ++index)
    {
        if (index > 0) out << ',';
        out << word_exit_signed_errors_ms[index];
    }
    out << "],\n"
        << "  \"word_interval_coverages\": [";
    for (size_t index = 0; index < word_interval_coverages.size(); ++index)
    {
        if (index > 0) out << ',';
        out << word_interval_coverages[index];
    }
    out << "],\n"
        << "  \"word_interval_precisions\": [";
    for (size_t index = 0; index < word_interval_precisions.size(); ++index)
    {
        if (index > 0) out << ',';
        out << word_interval_precisions[index];
    }
    out << "],\n"
        << "  \"word_interval_ious\": [";
    for (size_t index = 0; index < word_interval_ious.size(); ++index)
    {
        if (index > 0) out << ',';
        out << word_interval_ious[index];
    }
    out << "],\n"
        << "  \"word_duration_ratios\": [";
    for (size_t index = 0; index < word_duration_ratios.size(); ++index)
    {
        if (index > 0) out << ',';
        out << word_duration_ratios[index];
    }
    out << "],\n"
        << "  \"word_duration_abs_errors_ms\": [";
    for (size_t index = 0; index < word_duration_abs_errors_ms.size(); ++index)
    {
        if (index > 0) out << ',';
        out << word_duration_abs_errors_ms[index];
    }
    out << "],\n"
        << "  \"word_duration_signed_errors_ms\": [";
    for (size_t index = 0; index < word_duration_signed_errors_ms.size(); ++index)
    {
        if (index > 0) out << ',';
        out << word_duration_signed_errors_ms[index];
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

struct PhoneticPresentationReport
{
    std::string Csv;
    std::string Json;
};

static PhoneticPresentationReport phonetic_presentation_report(
    const FOffgridAIAlignedVisemeTrack& track,
    const std::vector<HandmadeLabel>& gold_visible)
{
    constexpr double SampleSeconds = 0.010;
    constexpr double VisibleWeight = 0.40;
    constexpr double SaturatedWeight = 0.90;
    const auto is_vowel = [](const std::string& phone) {
        static const std::vector<std::string> vowels = {
            "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER",
            "EY", "IH", "IY", "OW", "OY", "UH", "UW"};
        return std::find(vowels.begin(), vowels.end(), phone) != vowels.end();
    };
    const auto is_bilabial = [](const std::string& phone) {
        return phone == "M" || phone == "B" || phone == "P";
    };

    std::map<int, const FOffgridAIAlignedVisemeEvent*> events;
    for (const auto& event : track.Events)
    {
        if (!event.bIsRenderable || event.bCanceledByWordHandoff) continue;
        events[event.EventIndex] = &event;
    }
    struct Peak
    {
        double Weight = 0.0;
        double Seconds = 0.0;
    };
    std::map<int, Peak> peaks;
    double track_end = std::max(0.0, static_cast<double>(track.SpeechEndSeconds));
    for (const auto& event : track.Events)
        track_end = std::max(track_end, static_cast<double>(event.RenderEndSeconds));
    for (double seconds = 0.0; seconds <= track_end + 0.0001; seconds += SampleSeconds)
    {
        const auto samples = FOffgridAIVisemePerformer::Sample(
            track, static_cast<float>(seconds), true);
        for (const auto& sample : samples)
        {
            const auto event = events.find(sample.EventIndex);
            if (event == events.end()
                || seconds + 0.0005 < event->second->CommitPlaybackSeconds)
                continue;
            Peak& peak = peaks[sample.EventIndex];
            if (sample.SubmittedWeight > peak.Weight)
            {
                peak.Weight = sample.SubmittedWeight;
                peak.Seconds = seconds;
            }
        }
    }

    int vowel_labels = 0;
    int supported_vowels = 0;
    int vowel_frames = 0;
    int vowel_target_dominant_frames = 0;
    int vowel_same_word_dominant_frames = 0;
    int vowel_foreign_word_dominant_frames = 0;
    int low_dominance_vowels = 0;
    int foreign_intrusion_vowels = 0;
    int bilabial_labels = 0;
    int supported_bilabials = 0;
    int bilabial_frames = 0;
    int bilabial_target_dominant_frames = 0;
    int late_bilabial_peaks = 0;
    std::vector<double> bilabial_peak_errors_ms;
    int oh_frames = 0;
    int oh_saturated_frames = 0;
    int other_vowel_frames = 0;
    int other_vowel_saturated_frames = 0;

    std::ostringstream csv;
    csv << "word_index,word,phone,pose,gold_start_sec,gold_end_sec,target_event_index,"
           "target_peak_weight,target_peak_sec,target_peak_error_ms,total_frames,"
           "target_visible_frames,target_saturated_frames,target_dominant_frames,"
           "same_word_dominant_frames,foreign_word_dominant_frames,"
           "target_visible_rate,target_saturation_rate,target_dominance_rate,"
           "same_word_dominance_rate,foreign_word_dominance_rate,is_vowel,is_bilabial\n";
    csv << std::fixed << std::setprecision(6);

    for (const HandmadeLabel& label : gold_visible)
    {
        const bool vowel = is_vowel(label.source_phone_base);
        const bool bilabial = is_bilabial(label.source_phone_base);
        if (!vowel && !bilabial) continue;
        if (vowel) ++vowel_labels;
        if (bilabial) ++bilabial_labels;
        const auto target = std::find_if(track.Events.begin(), track.Events.end(),
            [&](const FOffgridAIAlignedVisemeEvent& event) {
                return event.bIsRenderable && !event.bCanceledByWordHandoff
                    && event.WordIndex == label.word_index
                    && event.SourcePhoneIndex == label.source_phone_global_index
                    && to_std(event.PoseID) == label.pose;
            });
        if (target == track.Events.end()) continue;
        if (vowel) ++supported_vowels;
        if (bilabial) ++supported_bilabials;

        int total_frames = 0;
        int target_visible_frames = 0;
        int target_saturated_frames = 0;
        int target_dominant_frames = 0;
        int same_word_dominant_frames = 0;
        int foreign_word_dominant_frames = 0;
        for (double seconds = label.start + 0.5 * SampleSeconds;
             seconds < label.end; seconds += SampleSeconds)
        {
            ++total_frames;
            double target_weight = 0.0;
            double dominant_weight = 0.0;
            const FOffgridAIAlignedVisemeEvent* dominant = nullptr;
            const auto samples = FOffgridAIVisemePerformer::Sample(
                track, static_cast<float>(seconds), true);
            for (const auto& sample : samples)
            {
                const auto event = events.find(sample.EventIndex);
                if (event == events.end()
                    || seconds + 0.0005 < event->second->CommitPlaybackSeconds)
                    continue;
                if (sample.EventIndex == target->EventIndex)
                    target_weight = std::max(
                        target_weight, static_cast<double>(sample.SubmittedWeight));
                if (sample.SubmittedWeight > dominant_weight)
                {
                    dominant_weight = sample.SubmittedWeight;
                    dominant = event->second;
                }
            }
            target_visible_frames += target_weight >= VisibleWeight;
            target_saturated_frames += target_weight >= SaturatedWeight;
            if (dominant && dominant_weight >= 0.20)
            {
                target_dominant_frames += dominant->EventIndex == target->EventIndex;
                same_word_dominant_frames += dominant->WordIndex == label.word_index;
                foreign_word_dominant_frames += dominant->WordIndex != label.word_index;
            }
        }
        const auto frame_rate = [&](int frames) {
            return total_frames > 0
                ? static_cast<double>(frames) / total_frames : 0.0;
        };
        const Peak peak = peaks[target->EventIndex];
        const double gold_center = 0.5 * (label.start + label.end);
        const double peak_error_ms = (peak.Seconds - gold_center) * 1000.0;
        if (vowel)
        {
            vowel_frames += total_frames;
            vowel_target_dominant_frames += target_dominant_frames;
            vowel_same_word_dominant_frames += same_word_dominant_frames;
            vowel_foreign_word_dominant_frames += foreign_word_dominant_frames;
            low_dominance_vowels += frame_rate(target_dominant_frames) < 0.50;
            foreign_intrusion_vowels += frame_rate(foreign_word_dominant_frames) > 0.25;
            if (label.source_phone_base == "OW")
            {
                oh_frames += total_frames;
                oh_saturated_frames += target_saturated_frames;
            }
            else
            {
                other_vowel_frames += total_frames;
                other_vowel_saturated_frames += target_saturated_frames;
            }
        }
        if (bilabial)
        {
            bilabial_frames += total_frames;
            bilabial_target_dominant_frames += target_dominant_frames;
            bilabial_peak_errors_ms.push_back(std::abs(peak_error_ms));
            late_bilabial_peaks += peak_error_ms > 20.0;
        }
        csv << label.word_index << ',' << csv_value(label.word) << ','
            << label.source_phone_base << ',' << label.pose << ','
            << label.start << ',' << label.end << ',' << target->EventIndex << ','
            << peak.Weight << ',' << peak.Seconds << ',' << peak_error_ms << ','
            << total_frames << ',' << target_visible_frames << ','
            << target_saturated_frames << ',' << target_dominant_frames << ','
            << same_word_dominant_frames << ',' << foreign_word_dominant_frames << ','
            << frame_rate(target_visible_frames) << ','
            << frame_rate(target_saturated_frames) << ','
            << frame_rate(target_dominant_frames) << ','
            << frame_rate(same_word_dominant_frames) << ','
            << frame_rate(foreign_word_dominant_frames) << ','
            << (vowel ? 1 : 0) << ',' << (bilabial ? 1 : 0) << '\n';
    }

    const auto rate = [](int numerator, int denominator) {
        return denominator > 0
            ? static_cast<double>(numerator) / denominator : 1.0;
    };
    std::ostringstream json;
    json << std::fixed << std::setprecision(6)
        << "{\n"
        << "  \"vowel_labels\": " << vowel_labels << ",\n"
        << "  \"supported_vowels\": " << supported_vowels << ",\n"
        << "  \"vowel_frames\": " << vowel_frames << ",\n"
        << "  \"vowel_target_dominant_frames\": " << vowel_target_dominant_frames << ",\n"
        << "  \"vowel_same_word_dominant_frames\": " << vowel_same_word_dominant_frames << ",\n"
        << "  \"vowel_foreign_word_dominant_frames\": " << vowel_foreign_word_dominant_frames << ",\n"
        << "  \"vowel_target_dominance_rate\": "
        << rate(vowel_target_dominant_frames, vowel_frames) << ",\n"
        << "  \"vowel_same_word_dominance_rate\": "
        << rate(vowel_same_word_dominant_frames, vowel_frames) << ",\n"
        << "  \"vowel_foreign_word_dominance_rate\": "
        << rate(vowel_foreign_word_dominant_frames, vowel_frames) << ",\n"
        << "  \"low_dominance_vowels\": " << low_dominance_vowels << ",\n"
        << "  \"foreign_intrusion_vowels\": " << foreign_intrusion_vowels << ",\n"
        << "  \"bilabial_labels\": " << bilabial_labels << ",\n"
        << "  \"supported_bilabials\": " << supported_bilabials << ",\n"
        << "  \"bilabial_frames\": " << bilabial_frames << ",\n"
        << "  \"bilabial_target_dominant_frames\": "
        << bilabial_target_dominant_frames << ",\n"
        << "  \"bilabial_target_dominance_rate\": "
        << rate(bilabial_target_dominant_frames, bilabial_frames) << ",\n"
        << "  \"late_bilabial_peaks\": " << late_bilabial_peaks << ",\n"
        << "  \"bilabial_peak_abs_errors_ms\": [";
    for (size_t index = 0; index < bilabial_peak_errors_ms.size(); ++index)
    {
        if (index > 0) json << ',';
        json << bilabial_peak_errors_ms[index];
    }
    json << "],\n"
        << "  \"oh_frames\": " << oh_frames << ",\n"
        << "  \"oh_saturated_frames\": " << oh_saturated_frames << ",\n"
        << "  \"oh_saturation_rate\": " << rate(oh_saturated_frames, oh_frames) << ",\n"
        << "  \"other_vowel_frames\": " << other_vowel_frames << ",\n"
        << "  \"other_vowel_saturated_frames\": " << other_vowel_saturated_frames << ",\n"
        << "  \"other_vowel_saturation_rate\": "
        << rate(other_vowel_saturated_frames, other_vowel_frames) << "\n"
        << "}\n";
    return {csv.str(), json.str()};
}

static std::string viseme_proportion_grade_json(
    const FOffgridAIAlignedVisemeTrack& track,
    const FOffgridAITextVisemePlan& plan,
    const std::vector<HandmadeLabel>& gold_visible,
    std::string* out_run_csv)
{
    constexpr double SampleSeconds = 0.010;
    constexpr double MinimumVisibleWeight = 0.020;

    struct Run
    {
        int WordIndex = -1;
        int RunIndexInWord = -1;
        std::string Word;
        std::string Pose;
        std::vector<int> EventIndices;
        double GoldDurationSec = 0.0;
        double RuntimeDurationSec = 0.0;
    };

    // Attach MFA duration to the transcript event that owns the matching
    // phone. Pronunciation insertions/deletions remain visible as zero-target
    // or zero-runtime mass instead of silently disappearing from the score.
    std::map<std::tuple<int, int, std::string>, double> gold_duration_by_phone;
    for (const HandmadeLabel& label : gold_visible)
    {
        gold_duration_by_phone[{
            label.word_index,
            label.source_phone_global_index,
            label.pose}] += std::max(0.0, label.end - label.start);
    }

    std::vector<Run> runs;
    std::map<int, int> run_by_event;
    std::map<int, std::vector<int>> runs_by_word;
    for (int event_index = 0; event_index < plan.Events.Num(); ++event_index)
    {
        const FOffgridAITextVisemeEvent& event = plan.Events[event_index];
        if (!event.bIsRenderable || event.Strength <= 0.012f) continue;
        const std::string pose = to_std(event.PoseID);
        int run_index = -1;
        auto& word_runs = runs_by_word[event.WordIndex];
        if (!word_runs.empty())
        {
            Run& previous = runs[static_cast<size_t>(word_runs.back())];
            if (previous.Pose == pose) run_index = word_runs.back();
        }
        if (run_index < 0)
        {
            Run run;
            run.WordIndex = event.WordIndex;
            run.RunIndexInWord = static_cast<int>(word_runs.size());
            run.Word = to_std(event.SourceText);
            run.Pose = pose;
            runs.push_back(std::move(run));
            run_index = static_cast<int>(runs.size()) - 1;
            word_runs.push_back(run_index);
        }
        Run& run = runs[static_cast<size_t>(run_index)];
        run.EventIndices.push_back(event_index);
        const auto gold = gold_duration_by_phone.find({
            event.WordIndex,
            event.SourcePhoneGlobalIndex,
            pose});
        if (gold != gold_duration_by_phone.end())
            run.GoldDurationSec += gold->second;
        run_by_event[event_index] = run_index;
    }

    // Measure what the face actually presents. At every runtime sample, the
    // strongest event in each word owns that 10 ms slice. This naturally
    // includes blend envelopes while preventing overlaps from double-counting
    // word duration.
    double track_end = std::max(0.0, static_cast<double>(track.SpeechEndSeconds));
    for (const auto& event : track.Events)
        track_end = std::max(track_end, static_cast<double>(event.RenderEndSeconds));
    std::map<int, const FOffgridAIAlignedVisemeEvent*> committed_by_event;
    for (const auto& event : track.Events)
    {
        if (!event.bIsRenderable || event.bCanceledByWordHandoff
            || event.Strength <= 0.012f) continue;
        committed_by_event[event.EventIndex] = &event;
    }
    for (double seconds = 0.0; seconds <= track_end + 0.0001; seconds += SampleSeconds)
    {
        std::map<int, std::pair<double, int>> winner_by_word;
        const auto samples = FOffgridAIVisemePerformer::Sample(
            track, static_cast<float>(seconds), true);
        for (const auto& sample : samples)
        {
            if (sample.SubmittedWeight < MinimumVisibleWeight) continue;
            const auto committed = committed_by_event.find(sample.EventIndex);
            const auto run = run_by_event.find(sample.EventIndex);
            if (committed == committed_by_event.end() || run == run_by_event.end()) continue;
            if (seconds + 0.0005 < committed->second->CommitPlaybackSeconds) continue;
            const int word_index = runs[static_cast<size_t>(run->second)].WordIndex;
            auto& winner = winner_by_word[word_index];
            if (sample.SubmittedWeight > winner.first)
                winner = {sample.SubmittedWeight, run->second};
        }
        for (const auto& [word_index, winner] : winner_by_word)
        {
            (void)word_index;
            runs[static_cast<size_t>(winner.second)].RuntimeDurationSec += SampleSeconds;
        }
    }

    int planned_multi_run_words = 0;
    int gradeable_words = 0;
    int ungradeable_words = 0;
    int zero_runtime_words = 0;
    int severe_words = 0;
    int expected_runs = 0;
    int runs_within_10pp = 0;
    std::vector<double> run_errors;
    std::vector<double> boundary_errors;
    std::vector<double> word_total_variations;
    std::ostringstream csv;
    csv << "word_index,word,run_index,pose,gold_duration_ms,runtime_duration_ms,"
           "gold_share,runtime_share,absolute_share_error,word_total_variation,gradeable\n";
    csv << std::fixed << std::setprecision(6);

    for (const auto& [word_index, word_run_indices] : runs_by_word)
    {
        if (word_run_indices.size() < 2) continue;
        ++planned_multi_run_words;
        double gold_total = 0.0;
        double runtime_total = 0.0;
        int gold_supported_runs = 0;
        for (int run_index : word_run_indices)
        {
            const Run& run = runs[static_cast<size_t>(run_index)];
            gold_total += run.GoldDurationSec;
            runtime_total += run.RuntimeDurationSec;
            if (run.GoldDurationSec > 0.0) ++gold_supported_runs;
        }
        const bool gradeable = gold_total > 0.0 && gold_supported_runs >= 2;
        if (!gradeable)
        {
            ++ungradeable_words;
            continue;
        }
        ++gradeable_words;
        if (runtime_total <= 0.0) ++zero_runtime_words;

        std::vector<double> gold_shares;
        std::vector<double> runtime_shares;
        double absolute_sum = 0.0;
        for (int run_index : word_run_indices)
        {
            const Run& run = runs[static_cast<size_t>(run_index)];
            const double gold_share = run.GoldDurationSec / gold_total;
            const double runtime_share = runtime_total > 0.0
                ? run.RuntimeDurationSec / runtime_total : 0.0;
            gold_shares.push_back(gold_share);
            runtime_shares.push_back(runtime_share);
            absolute_sum += std::abs(runtime_share - gold_share);
        }
        const double total_variation = 0.5 * absolute_sum;
        word_total_variations.push_back(total_variation);
        if (total_variation > 0.25) ++severe_words;

        double gold_cumulative = 0.0;
        double runtime_cumulative = 0.0;
        for (size_t local = 0; local < word_run_indices.size(); ++local)
        {
            const Run& run = runs[static_cast<size_t>(word_run_indices[local])];
            const double error = std::abs(runtime_shares[local] - gold_shares[local]);
            run_errors.push_back(error);
            ++expected_runs;
            if (error <= 0.10) ++runs_within_10pp;
            if (local + 1 < word_run_indices.size())
            {
                gold_cumulative += gold_shares[local];
                runtime_cumulative += runtime_shares[local];
                boundary_errors.push_back(std::abs(runtime_cumulative - gold_cumulative));
            }
            csv << word_index << ",\"" << json_escape(run.Word) << "\"," << local
                << ',' << run.Pose << ',' << run.GoldDurationSec * 1000.0
                << ',' << run.RuntimeDurationSec * 1000.0 << ',' << gold_shares[local]
                << ',' << runtime_shares[local] << ',' << error << ','
                << total_variation << ",1\n";
        }
    }
    if (out_run_csv) *out_run_csv = csv.str();

    const auto rate = [](int numerator, int denominator) {
        return denominator > 0
            ? static_cast<double>(numerator) / static_cast<double>(denominator)
            : 1.0;
    };
    std::ostringstream out;
    out << std::fixed << std::setprecision(6)
        << "{\n"
        << "  \"planned_multi_viseme_words\": " << planned_multi_run_words << ",\n"
        << "  \"gradeable_words\": " << gradeable_words << ",\n"
        << "  \"ungradeable_words\": " << ungradeable_words << ",\n"
        << "  \"word_coverage_rate\": " << rate(gradeable_words, planned_multi_run_words) << ",\n"
        << "  \"zero_runtime_words\": " << zero_runtime_words << ",\n"
        << "  \"expected_viseme_runs\": " << expected_runs << ",\n"
        << "  \"runs_within_10pp\": " << runs_within_10pp << ",\n"
        << "  \"runs_within_10pp_rate\": " << rate(runs_within_10pp, expected_runs) << ",\n"
        << "  \"run_share_abs_error_mean\": " << mean_ms(run_errors) << ",\n"
        << "  \"run_share_abs_error_median\": " << percentile_ms(run_errors, 0.5) << ",\n"
        << "  \"run_share_abs_error_p90\": " << percentile_ms(run_errors, 0.9) << ",\n"
        << "  \"boundary_position_abs_error_mean\": " << mean_ms(boundary_errors) << ",\n"
        << "  \"boundary_position_abs_error_median\": " << percentile_ms(boundary_errors, 0.5) << ",\n"
        << "  \"boundary_position_abs_error_p90\": " << percentile_ms(boundary_errors, 0.9) << ",\n"
        << "  \"word_total_variation_mean\": " << mean_ms(word_total_variations) << ",\n"
        << "  \"word_total_variation_median\": " << percentile_ms(word_total_variations, 0.5) << ",\n"
        << "  \"word_total_variation_p90\": " << percentile_ms(word_total_variations, 0.9) << ",\n"
        << "  \"severe_words\": " << severe_words << ",\n"
        << "  \"run_share_abs_errors\": [";
    for (size_t index = 0; index < run_errors.size(); ++index)
        out << (index > 0 ? "," : "") << run_errors[index];
    out << "],\n  \"boundary_position_abs_errors\": [";
    for (size_t index = 0; index < boundary_errors.size(); ++index)
        out << (index > 0 ? "," : "") << boundary_errors[index];
    out << "],\n  \"word_total_variations\": [";
    for (size_t index = 0; index < word_total_variations.size(); ++index)
        out << (index > 0 ? "," : "") << word_total_variations[index];
    out << "]\n}\n";
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
    std::map<std::string, double> previous_displayed_weights;
    std::map<std::string, double> previous_displayed_velocity;
    double squared_step_sum = 0.0;
    double squared_acceleration_sum = 0.0;
    double max_displayed_pose_step = 0.0;
    double max_displayed_pose_speed = 0.0;
    double max_displayed_pose_acceleration = 0.0;
    std::map<std::string, double> previous_driver_weights;
    std::map<std::string, double> previous_driver_velocity;
    double driver_squared_step_sum = 0.0;
    double driver_squared_acceleration_sum = 0.0;
    double driver_max_pose_step = 0.0;
    double driver_max_pose_speed = 0.0;
    double driver_max_pose_acceleration = 0.0;
    int driver_channel_sample_count = 0;
    int sample_count = 0;
    int displayed_pose_sample_count = 0;
    std::map<std::string, bool> eligible_pose_ids;
    std::map<std::string, bool> driven_pose_ids;
    for (const auto& event : track.Events)
    {
        if (event.bIsRenderable && !event.bCanceledByWordHandoff
            && !event.PoseID.IsNone())
            eligible_pose_ids[to_std(event.PoseID)] = true;
    }
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
            FOffgridAIVisemePerformer::BuildPoseStateFromSamples(samples);
        for (const auto& sample : samples)
        {
            if (!sample.PoseID.IsNone()
                && !target.PoseWeights.Contains(sample.PoseID))
                throw std::runtime_error(
                    "detailed pose was filtered from performer target: "
                    + to_std(sample.PoseID));
        }
        displayed = FOffgridAIVisemePerformer::StepDisplayedPose(
            displayed,
            target,
            static_cast<float>(frame_seconds),
            0.035f,
            false);
        TMap<FName, float> displayed_pose_map;
        FOffgridAIVisemePerformer::BuildPoseWeightMapFromState(
            displayed_pose_map, displayed);
        for (const FName& pose_id : target.PoseIDs)
        {
            if (!displayed_pose_map.Contains(pose_id))
                throw std::runtime_error(
                    "detailed pose was filtered from displayed state: "
                    + to_std(pose_id));
        }
        std::map<std::string, double> displayed_weights;
        for (const auto& pose : eligible_pose_ids)
        {
            const FName pose_id(pose.first.c_str());
            const double weight = displayed_pose_map.FindRef(pose_id);
            displayed_weights[pose.first] = weight;
            if (weight > 0.012) driven_pose_ids[pose.first] = true;
        }
        if (sample_count > 0)
        {
            for (const auto& pose : eligible_pose_ids)
            {
                const double current = displayed_weights.count(pose.first)
                    ? displayed_weights.at(pose.first) : 0.0;
                const double previous = previous_displayed_weights.count(pose.first)
                    ? previous_displayed_weights.at(pose.first) : 0.0;
                const double step = current - previous;
                const double velocity = step / frame_seconds;
                const double previous_velocity =
                    previous_displayed_velocity.count(pose.first)
                    ? previous_displayed_velocity.at(pose.first) : 0.0;
                const double acceleration =
                    (velocity - previous_velocity) / frame_seconds;
                max_displayed_pose_step = std::max(
                    max_displayed_pose_step, std::abs(step));
                max_displayed_pose_speed = std::max(
                    max_displayed_pose_speed, std::abs(velocity));
                max_displayed_pose_acceleration = std::max(
                    max_displayed_pose_acceleration, std::abs(acceleration));
                squared_step_sum += step * step;
                squared_acceleration_sum += acceleration * acceleration;
                previous_displayed_velocity[pose.first] = velocity;
                ++displayed_pose_sample_count;
            }
        }
        previous_displayed_weights = displayed_weights;
        ++sample_count;
    }

    const double rms_displayed_pose_step = displayed_pose_sample_count > 0
        ? std::sqrt(squared_step_sum
            / static_cast<double>(displayed_pose_sample_count))
        : 0.0;
    const double rms_displayed_pose_acceleration = displayed_pose_sample_count > 0
        ? std::sqrt(squared_acceleration_sum
            / static_cast<double>(displayed_pose_sample_count))
        : 0.0;
    const double driver_rms_pose_step = driver_channel_sample_count > 0
        ? std::sqrt(driver_squared_step_sum
            / static_cast<double>(driver_channel_sample_count))
        : 0.0;
    const double driver_rms_pose_acceleration = driver_channel_sample_count > 0
        ? std::sqrt(driver_squared_acceleration_sum
            / static_cast<double>(driver_channel_sample_count))
        : 0.0;
    std::ostringstream out;
    out << std::fixed << std::setprecision(6)
        << "{\n"
        << "  \"sample_rate_hz\": 60.0,\n"
        << "  \"sample_count\": " << sample_count << ",\n"
        << "  \"eligible_detailed_pose_count\": "
        << eligible_pose_ids.size() << ",\n"
        << "  \"driven_detailed_pose_count\": "
        << driven_pose_ids.size() << ",\n"
        << "  \"displayed_max_pose_step_per_frame\": "
        << max_displayed_pose_step << ",\n"
        << "  \"displayed_rms_pose_step_per_frame\": "
        << rms_displayed_pose_step << ",\n"
        << "  \"displayed_max_pose_speed_per_second\": "
        << max_displayed_pose_speed << ",\n"
        << "  \"displayed_max_pose_acceleration_per_second2\": "
        << max_displayed_pose_acceleration << ",\n"
        << "  \"displayed_rms_pose_acceleration_per_second2\": "
        << rms_displayed_pose_acceleration << ",\n"
        << "  \"driver_max_pose_step_per_frame\": "
        << driver_max_pose_step << ",\n"
        << "  \"driver_rms_pose_step_per_frame\": "
        << driver_rms_pose_step << ",\n"
        << "  \"driver_max_pose_speed_per_second\": "
        << driver_max_pose_speed << ",\n"
        << "  \"driver_max_pose_acceleration_per_second2\": "
        << driver_max_pose_acceleration << ",\n"
        << "  \"driver_rms_pose_acceleration_per_second2\": "
        << driver_rms_pose_acceleration << "\n"
        << "}\n";
    return out.str();
}

struct PresentationVisibilityReport
{
    std::string Csv;
    std::string Json;
};

static PresentationVisibilityReport presentation_event_visibility_report(
    const FOffgridAIAlignedVisemeTrack& track)
{
    constexpr double frame_seconds = 1.0 / 30.0;
    constexpr int phase_count = 3;
    constexpr float visible_weight = 0.20f;
    constexpr int minimum_visible_frames = 2;

    struct PhaseStats
    {
        double peak = 0.0;
        int frames_above = 0;
    };
    struct EventStats
    {
        const FOffgridAICommittedVisemeEvent* event = nullptr;
        std::array<PhaseStats, phase_count> phases{};
        double region_start_offset_ms = 0.0;
    };

    std::map<int32, EventStats> stats;
    double end_seconds = std::max(0.0, static_cast<double>(track.SpeechEndSeconds));
    for (const auto& event : track.Events)
    {
        if (!event.bIsRenderable || event.bCanceledByWordHandoff
            || event.PoseID.IsNone())
            continue;
        EventStats row;
        row.event = &event;
        for (const auto& region : track.SpeechRegions)
        {
            if (region.SpeechRegionIndex != event.SpeechRegionIndex) continue;
            row.region_start_offset_ms = 1000.0 * (
                static_cast<double>(event.FinalRenderCenterSeconds)
                - static_cast<double>(region.StartSeconds));
            break;
        }
        stats[event.EventIndex] = row;
        end_seconds = std::max(
            end_seconds,
            static_cast<double>(event.FinalRenderCenterSeconds) + 0.300);
    }

    for (int phase = 0; phase < phase_count; ++phase)
    {
        FOffgridAILipsyncPoseRuntimeState displayed;
        const double phase_seconds = phase * frame_seconds / phase_count;
        for (double seconds = phase_seconds; seconds <= end_seconds + 0.0001;
             seconds += frame_seconds)
        {
            const auto samples = FOffgridAIVisemePerformer::Sample(
                track, static_cast<float>(seconds), true);
            const auto target =
                FOffgridAIVisemePerformer::BuildPoseStateFromSamples(samples);
            displayed = FOffgridAIVisemePerformer::StepDisplayedPose(
                displayed,
                target,
                static_cast<float>(frame_seconds),
                0.035f,
                false);
            TMap<FName, float> displayed_pose_map;
            FOffgridAIVisemePerformer::BuildPoseWeightMapFromState(
                displayed_pose_map, displayed);
            for (const auto& sample : samples)
            {
                auto found = stats.find(sample.EventIndex);
                if (found == stats.end()) continue;
                const double weight = displayed_pose_map.FindRef(sample.PoseID);
                PhaseStats& phase_stats = found->second.phases[phase];
                phase_stats.peak = std::max(phase_stats.peak, weight);
                if (weight >= visible_weight) ++phase_stats.frames_above;
            }
        }
    }

    int robust_visible = 0;
    int boundary_events = 0;
    int robust_boundary_events = 0;
    std::ostringstream csv;
    csv << "event_index,pose,word,center,region_start_offset_ms,"
           "min_peak_face_driver,min_frames_above_020,min_ms_above_020,"
           "visible_phase_count,robust_visible\n";
    for (const auto& pair : stats)
    {
        const EventStats& row = pair.second;
        double minimum_peak = std::numeric_limits<double>::infinity();
        int minimum_frames = std::numeric_limits<int>::max();
        int visible_phases = 0;
        for (const PhaseStats& phase : row.phases)
        {
            minimum_peak = std::min(minimum_peak, phase.peak);
            minimum_frames = std::min(minimum_frames, phase.frames_above);
            if (phase.peak >= visible_weight
                && phase.frames_above >= minimum_visible_frames)
                ++visible_phases;
        }
        const bool is_robust = visible_phases == phase_count;
        if (is_robust) ++robust_visible;
        const bool is_boundary = row.region_start_offset_ms <= 60.0;
        if (is_boundary)
        {
            ++boundary_events;
            if (is_robust) ++robust_boundary_events;
        }
        csv << pair.first << ','
            << to_std(row.event->PoseID) << ','
            << csv_value(to_std(row.event->SourceWord)) << ','
            << std::fixed << std::setprecision(6)
            << row.event->FinalRenderCenterSeconds << ','
            << std::setprecision(3) << row.region_start_offset_ms << ','
            << std::setprecision(6) << minimum_peak << ','
            << minimum_frames << ','
            << std::setprecision(3) << minimum_frames * frame_seconds * 1000.0 << ','
            << visible_phases << ',' << (is_robust ? 1 : 0) << '\n';
    }

    std::ostringstream json;
    json << std::fixed << std::setprecision(6)
        << "{\n"
        << "  \"sample_rate_hz\": 30.0,\n"
        << "  \"phase_count\": " << phase_count << ",\n"
        << "  \"eligible_event_count\": " << stats.size() << ",\n"
        << "  \"robust_visible_event_count\": " << robust_visible << ",\n"
        << "  \"robust_visible_event_rate\": "
        << (stats.empty() ? 1.0
            : static_cast<double>(robust_visible) / stats.size()) << ",\n"
        << "  \"boundary_event_count\": " << boundary_events << ",\n"
        << "  \"robust_boundary_event_count\": " << robust_boundary_events << ",\n"
        << "  \"robust_boundary_event_rate\": "
        << (boundary_events == 0 ? 1.0
            : static_cast<double>(robust_boundary_events) / boundary_events) << "\n"
        << "}\n";
    return {csv.str(), json.str()};
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
        const fs::path root = cli.root;
        const StreamConfig stream = cli.stream;
        const OutputConfig output = cli.output;
        fs::path out_root = root / "outputs" / "runs" / "latest";
        fs::remove_all(out_root);
        fs::create_directories(out_root);

        std::vector<fs::path> transcript_paths = read_corpus_transcript_paths(root);
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
            const auto visibility = presentation_event_visibility_report(committed);
            write_text(case_dir / "presentation_event_visibility.csv", visibility.Csv);
            write_text(case_dir / "presentation_event_visibility_grade.json", visibility.Json);
            if (output.write_detailed_diagnostics)
            {
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
                if (!cli.case_filter.empty()) std::cerr << "[case] visible-labels " << stem << std::endl;
                const auto handmade = build_gold_visible_labels(plan, gold_phones);
                gold_viseme_count = handmade.size();
                if (output.write_monotonic_training_rows)
                {
                    write_text(
                        case_dir / "monotonic_audio_frames.csv",
                        monotonic_audio_frames_csv(
                            session.GetAudioFeatureExtractor().GetFeatureFrames()));
                    write_text(
                        case_dir / "monotonic_tokens.csv",
                        monotonic_tokens_csv(plan, handmade, gold_words));
                }
                if (!cli.case_filter.empty()) std::cerr << "[case] grade " << stem << std::endl;
                write_text(case_dir / "planned_words.csv", planned_words_csv(plan));
                write_text(case_dir / "planned_syllables.csv", planned_syllables_csv(plan));
                write_text(case_dir / "planned_boundaries.csv", planned_boundaries_csv(plan));
                write_text(case_dir / "gold_visible_visemes.csv", gold_visible_visemes_csv(handmade));
                write_text(case_dir / "focus_alignment_grade.json",
                    focus_alignment_grade_json(committed, plan, handmade, gold_speech));
                std::string runtime_word_delivery_csv;
                write_text(case_dir / "runtime_delivery_grade.json",
                    runtime_delivery_grade_json(
                        committed,
                        plan,
                        gold_words,
                        gold_speech,
                        &runtime_word_delivery_csv));
                write_text(
                    case_dir / "runtime_word_delivery.csv",
                    runtime_word_delivery_csv);
                const auto phonetic_presentation = phonetic_presentation_report(
                    committed, handmade);
                write_text(
                    case_dir / "phonetic_presentation.csv",
                    phonetic_presentation.Csv);
                write_text(
                    case_dir / "phonetic_presentation_grade.json",
                    phonetic_presentation.Json);
                std::string viseme_proportion_runs_csv;
                write_text(case_dir / "viseme_proportion_grade.json",
                    viseme_proportion_grade_json(
                        committed,
                        plan,
                        handmade,
                        &viseme_proportion_runs_csv));
                write_text(
                    case_dir / "viseme_proportion_runs.csv",
                    viseme_proportion_runs_csv);
                write_text(case_dir / "region_ownership_grade.json",
                    region_ownership_grade_json(committed, plan, gold_words, gold_speech));
                report = grade(committed, speech, handmade, gold_words, gold_speech);
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
