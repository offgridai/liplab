#include "Lipsync/OffgridAILipsyncRuntimeAdapter.h"
#include "Lipsync/OffgridAIVisemePerformer.h"

#include <cstring>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <iostream>
#include <optional>
#include <numeric>
#include <sstream>
#include <string>
#include <limits>
#include <vector>

namespace fs = std::filesystem;

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
    int phrase_index = -1;
    int sentence_index = -1;
};

struct GoldWordTiming
{
    int word_index = -1;
    std::string word;
    double start = 0.0;
    double end = 0.0;
    int phrase_index = -1;
    int sentence_index = -1;
};

struct GoldSpeechRegion
{
    int index = -1;
    double start = 0.0;
    double end = 0.0;
    double last_speech = 0.0;
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
    BoundaryAlignmentReport sentence_regions;
    BoundaryAlignmentReport clause_regions;
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

static std::string to_std(const FString& value) { return value.Str(); }
static std::string to_std(const FName& value) { return value.Str(); }

static std::vector<std::string> parse_csv_cells(const std::string& line)
{
    std::vector<std::string> cells;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ','))
    {
        cells.push_back(cell);
    }
    return cells;
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
    while (std::getline(file, line))
    {
        if (line.empty()) continue;
        const std::vector<std::string> cells = parse_csv_cells(line);
        if (cells.size() < 4) continue;
        HandmadeLabel label;
        label.start = std::stod(cells[0]);
        label.end = std::stod(cells[1]);
        label.pose = cells[2];
        label.word = cells[3];
        if (cells.size() > 4 && !cells[4].empty()) label.confidence = std::stod(cells[4]);
        if (cells.size() > 5 && !cells[5].empty()) label.word_index = std::stoi(cells[5]);
        if (cells.size() > 6 && !cells[6].empty()) label.phrase_index = std::stoi(cells[6]);
        if (cells.size() > 7 && !cells[7].empty()) label.sentence_index = std::stoi(cells[7]);
        out.push_back(label);
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
    while (std::getline(file, line))
    {
        if (line.empty()) continue;
        const std::vector<std::string> cells = parse_csv_cells(line);
        if (cells.size() < 4) continue;
        GoldWordTiming word;
        if (!cells[0].empty()) word.word_index = std::stoi(cells[0]);
        word.word = cells[1];
        word.start = std::stod(cells[2]);
        word.end = std::stod(cells[3]);
        if (cells.size() > 4 && !cells[4].empty()) word.phrase_index = std::stoi(cells[4]);
        if (cells.size() > 5 && !cells[5].empty()) word.sentence_index = std::stoi(cells[5]);
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
    while (std::getline(file, line))
    {
        if (line.empty()) continue;
        const std::vector<std::string> cells = parse_csv_cells(line);
        if (cells.size() < 3) continue;
        GoldSpeechRegion region;
        if (!cells[0].empty()) region.index = std::stoi(cells[0]);
        region.start = std::stod(cells[1]);
        region.end = std::stod(cells[2]);
        if (cells.size() > 3 && !cells[3].empty()) region.last_speech = std::stod(cells[3]);
        out.push_back(region);
    }
    return out;
}

static std::string planned_csv(const FOffgridAITextVisemePlan& plan)
{
    std::ostringstream out;
    out << "index,pose,word,word_index,phrase_index,sentence_index,text_center_norm,strength,source_phone,source_phone_index,generator\n";
    out << std::fixed << std::setprecision(6);
    for (int32 i = 0; i < plan.Events.Num(); ++i)
    {
        const auto& event = plan.Events[i];
        const double center = static_cast<double>(event.StartNorm + event.EndNorm) * 0.5;
        out << i << ','
            << to_std(event.PoseID) << ','
            << to_std(event.SourceText) << ','
            << event.WordIndex << ','
            << event.PhraseIndex << ','
            << event.SentenceIslandIndex << ','
            << center << ','
            << event.Strength << ','
            << to_std(event.SourcePhoneBase) << ','
            << event.SourcePhoneIndex << ','
            << to_std(event.Generator) << '\n';
    }
    return out.str();
}

static std::string speech_csv(const TArray<FOffgridAIStreamingSpeechIsland>& islands)
{
    std::ostringstream out;
    out << "index,start,end,last_speech,started,ended,provisional_end,end_decision,reopen_count,end_reason\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& island : islands)
    {
        out << island.IslandIndex << ','
            << island.AudioBufferStartSec << ','
            << island.AudioBufferEndSec << ','
            << island.AudioBufferLastSpeechSec << ','
            << (island.bStarted ? 1 : 0) << ','
            << (island.bEnded ? 1 : 0) << ','
            << island.ProvisionalEndSec << ','
            << island.EndDecisionSec << ','
            << island.ReopenCount << ','
            << to_std(island.EndReason) << '\n';
    }
    return out.str();
}

static std::string occupancy_frames_csv(const TArray<FOffgridAIStreamingAudioFeatureFrame>& frames)
{
    std::ostringstream out;
    out << "index,start,end,center,rms,rms_norm,delta_rms,flux,zcr,low_band,mid_band,high_band,centroid,periodicity,evidence,open_threshold,close_threshold,in_speech_before,in_speech_after,open_candidate,keep_open,strong_onset,strong_quiet,low_evidence,endpoint_active,silence_accum,endpoint_start,active_island_start,active_island_end,frame_started,frame_closed,frame_bridged,decision,local_rms_peak,local_rms_valley,local_flux_peak,pause_family,pause_family_confidence,pause_gap_age\n";
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
            << f.ActiveIslandStartSec << ','
            << f.ActiveIslandEndSec << ','
            << (f.bFrameStartedIsland ? 1 : 0) << ','
            << (f.bFrameClosedIsland ? 1 : 0) << ','
            << (f.bFrameBridgedIsland ? 1 : 0) << ','
            << to_std(f.OccupancyDecision) << ','
            << (f.bLocalRMSPeak ? 1 : 0) << ','
            << (f.bLocalRMSValley ? 1 : 0) << ','
            << (f.bLocalFluxPeak ? 1 : 0) << ','
            << to_std(f.PauseFamily) << ','
            << f.PauseFamilyConfidence << ','
            << f.PauseGapAgeSec << '\n';
    }
    return out.str();
}

static std::string gap_candidates_csv(const TArray<FOffgridAIStreamingSpeechGapCandidate>& gaps)
{
    std::ostringstream out;
    out << "gap_index,prev_island_index,next_island_index,gap_start,gap_end,gap_duration,prev_island_duration,quiet_evidence,quiet_rms_norm,reopen_evidence,reopen_flux,strong_quiet_close,strong_onset_reopen,bridged,close_reason,decision_class\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& gap : gaps)
    {
        out << gap.GapIndex << ','
            << gap.PrevIslandIndex << ','
            << gap.NextIslandIndex << ','
            << gap.GapStartSec << ','
            << gap.GapEndSec << ','
            << gap.GapDurationSec << ','
            << gap.PrevIslandDurationSec << ','
            << gap.QuietEvidence << ','
            << gap.QuietRMSNorm << ','
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

static std::string committed_csv(const FOffgridAIAlignedVisemeTrack& track)
{
    std::ostringstream out;
    out << "index,start,center,end,pose,word,word_index,phrase_index,sentence_index,strength,reason,source_phone_index,source_phone_base,source_phone_class,text_center_norm,text_diagnostic_center,commit_playback,commit_lead,required_active_elapsed,observed_active_elapsed,active_progress_deficit,required_progress_norm,observed_progress_norm,active_progress_ratio,mapped_to_observed_speech,detected_word_start,detected_word_start_mapped\n";
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
            << event.PhraseIndex << ','
            << event.SentenceIndex << ','
            << event.Strength << ','
            << to_std(event.CommitReason) << ','
            << event.SourcePhoneIndex << ','
            << to_std(event.SourcePhoneBase) << ','
            << to_std(event.SourcePhoneClass) << ','
            << event.TextCenterNorm << ','
            << event.TextDiagnosticCenterSeconds << ','
            << event.CommitPlaybackSeconds << ','
            << event.CommitLeadSeconds << ','
            << event.RequiredActiveElapsedSeconds << ','
            << event.ObservedActiveElapsedSeconds << ','
            << event.ActiveProgressDeficitSeconds << ','
            << event.RequiredProgressNorm << ','
            << event.ObservedProgressNorm << ','
            << event.ActiveProgressRatio << ','
            << (event.bMappedToObservedSpeech ? 1 : 0) << ','
            << event.DetectedWordStartSeconds << ','
            << (event.bDetectedWordStartMappedToObservedSpeech ? 1 : 0) << '\n';
    }
    return out.str();
}

static std::string commit_decisions_csv(const FOffgridAIAlignedVisemeTrack& track)
{
    std::ostringstream out;
    out << "event_index,pose,word,word_index,source_phone_index,source_phone_base,source_phone_class,commit_reason,render_start,render_center,render_end,commit_playback,commit_lead,required_active_elapsed,observed_active_elapsed,active_progress_deficit,required_progress_norm,observed_progress_norm,active_progress_ratio,mapped_to_observed_speech,detected_word_start,detected_word_start_mapped\n";
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
            << event.CommitPlaybackSeconds << ','
            << event.CommitLeadSeconds << ','
            << event.RequiredActiveElapsedSeconds << ','
            << event.ObservedActiveElapsedSeconds << ','
            << event.ActiveProgressDeficitSeconds << ','
            << event.RequiredProgressNorm << ','
            << event.ObservedProgressNorm << ','
            << event.ActiveProgressRatio << ','
            << (event.bMappedToObservedSpeech ? 1 : 0) << ','
            << event.DetectedWordStartSeconds << ','
            << (event.bDetectedWordStartMappedToObservedSpeech ? 1 : 0) << '\n';
    }
    return out.str();
}



static std::string stream_tail_csv(const FOffgridAIStreamTailDiagnosticRow& row)
{
    std::ostringstream out;
    out << "line_id,pcm_chunk_count,pcm_bytes_received,pcm_samples_received,last_sample_rate,last_num_channels,last_chunk_start_sample,last_chunk_end_sample,observed_audio_buffer_end,first_speech_audio_buffer_start,speech_island_count,input_stream_closed,diagnostic_kind\n";
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
        << row.SpeechIslandCount << ','
        << (row.bInputStreamClosed ? 1 : 0) << ','
        << to_std(row.DiagnosticKind) << '\n';
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
    if (gap_seconds >= 0.130) return "phrase_gap";
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
        const std::string family = to_std(frames[i].PauseFamily);
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
        const std::string pred_pause = to_std(frame.PauseFamily);
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
            << frame.PauseFamilyConfidence << ','
            << pause_match << ','
            << predicted_word_boundary << '\n';
    }

    return out.str();
}

static std::vector<TimeSpan> build_predicted_speech_regions(const TArray<FOffgridAIStreamingSpeechIsland>& islands)
{
    std::vector<TimeSpan> spans;
    for (const auto& island : islands)
    {
        const double start = static_cast<double>(island.AudioBufferStartSec);
        const double end = static_cast<double>(std::max(island.AudioBufferLastSpeechSec, island.AudioBufferEndSec));
        if (end <= start)
        {
            continue;
        }
        spans.push_back(TimeSpan{island.IslandIndex, start, end});
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
    int last_phrase = std::numeric_limits<int>::min();
    int synthetic_key = -1;
    for (const auto& word : words)
    {
        if (word.phrase_index < 0) continue;
        if (spans.empty() || word.sentence_index != last_sentence || word.phrase_index != last_phrase)
        {
            ++synthetic_key;
            spans.push_back(TimeSpan{synthetic_key, word.start, word.end});
            last_sentence = word.sentence_index;
            last_phrase = word.phrase_index;
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
    int last_phrase = std::numeric_limits<int>::min();
    int synthetic_key = -1;
    for (const auto& event : track.Events)
    {
        if (event.PhraseIndex < 0) continue;
        const double start = static_cast<double>(event.RenderStartSeconds);
        const double end = static_cast<double>(event.RenderEndSeconds);
        if (spans.empty() || event.SentenceIndex != last_sentence || event.PhraseIndex != last_phrase)
        {
            ++synthetic_key;
            spans.push_back(TimeSpan{synthetic_key, start, end});
            last_sentence = event.SentenceIndex;
            last_phrase = event.PhraseIndex;
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
        const double detected_start = event_it->bDetectedWordStartMappedToObservedSpeech
            ? static_cast<double>(event_it->DetectedWordStartSeconds)
            : static_cast<double>(event_it->RenderStartSeconds);
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

static GradeReport grade(
    const FOffgridAIAlignedVisemeTrack& track,
    const TArray<FOffgridAIStreamingSpeechIsland>& speech_islands,
    const std::vector<HandmadeLabel>& handmade,
    const std::vector<GoldWordTiming>& gold_words,
    const std::vector<GoldSpeechRegion>& gold_speech)
{
    GradeReport report;
    report.reference_count = static_cast<int>(handmade.size());
    report.committed_count = track.Events.Num();

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
        build_predicted_speech_regions(speech_islands));
    report.pause_alignment.sentence_regions = grade_region_boundaries(
        build_gold_sentence_regions(gold_words),
        build_predicted_sentence_regions(track));
    report.pause_alignment.clause_regions = grade_region_boundaries(
        build_gold_clause_regions(gold_words),
        build_predicted_clause_regions(track));
    report.word_onset_alignment = grade_word_onsets(track, handmade, gold_words);
    report.intra_word_alignment = grade_intra_word_alignment(track, handmade);
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
            const double detected_start = event_it->bDetectedWordStartMappedToObservedSpeech
                ? static_cast<double>(event_it->DetectedWordStartSeconds)
                : static_cast<double>(event_it->RenderStartSeconds);
            const double event_start = static_cast<double>(event_it->RenderStartSeconds);
            out << to_std(event_it->PoseID) << "," << event_start << ","
                << detected_start << ","
                << (event_it->bDetectedWordStartMappedToObservedSpeech ? 1 : 0) << ","
                << std::abs(detected_start - gold_start) * 1000.0 << ",false\n";
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
        << "    \"sentence_region_reference_count\": " << grade_report.pause_alignment.sentence_regions.reference_count << ",\n"
        << "    \"sentence_region_predicted_count\": " << grade_report.pause_alignment.sentence_regions.predicted_count << ",\n"
        << "    \"sentence_region_matched_count\": " << grade_report.pause_alignment.sentence_regions.matched_count << ",\n"
        << "    \"sentence_region_missing_count\": " << grade_report.pause_alignment.sentence_regions.missing_count << ",\n"
        << "    \"sentence_region_extra_count\": " << grade_report.pause_alignment.sentence_regions.extra_count << ",\n"
        << "    \"mean_abs_sentence_region_start_error_ms\": " << grade_report.pause_alignment.sentence_regions.mean_abs_start_error_ms << ",\n"
        << "    \"median_abs_sentence_region_start_error_ms\": " << grade_report.pause_alignment.sentence_regions.median_abs_start_error_ms << ",\n"
        << "    \"p90_abs_sentence_region_start_error_ms\": " << grade_report.pause_alignment.sentence_regions.p90_abs_start_error_ms << ",\n"
        << "    \"mean_abs_sentence_region_end_error_ms\": " << grade_report.pause_alignment.sentence_regions.mean_abs_end_error_ms << ",\n"
        << "    \"median_abs_sentence_region_end_error_ms\": " << grade_report.pause_alignment.sentence_regions.median_abs_end_error_ms << ",\n"
        << "    \"p90_abs_sentence_region_end_error_ms\": " << grade_report.pause_alignment.sentence_regions.p90_abs_end_error_ms << ",\n"
        << "    \"sentence_region_count_mismatch\": " << (grade_report.pause_alignment.sentence_regions.count_mismatch ? "true" : "false") << ",\n"
        << "    \"mean_sentence_region_lead_in_ms\": " << grade_report.pause_alignment.sentence_regions.mean_lead_in_ms << ",\n"
        << "    \"mean_sentence_region_tail_out_ms\": " << grade_report.pause_alignment.sentence_regions.mean_tail_out_ms << ",\n"
        << "    \"clause_region_reference_count\": " << grade_report.pause_alignment.clause_regions.reference_count << ",\n"
        << "    \"clause_region_predicted_count\": " << grade_report.pause_alignment.clause_regions.predicted_count << ",\n"
        << "    \"clause_region_matched_count\": " << grade_report.pause_alignment.clause_regions.matched_count << ",\n"
        << "    \"clause_region_missing_count\": " << grade_report.pause_alignment.clause_regions.missing_count << ",\n"
        << "    \"clause_region_extra_count\": " << grade_report.pause_alignment.clause_regions.extra_count << ",\n"
        << "    \"mean_abs_clause_region_start_error_ms\": " << grade_report.pause_alignment.clause_regions.mean_abs_start_error_ms << ",\n"
        << "    \"median_abs_clause_region_start_error_ms\": " << grade_report.pause_alignment.clause_regions.median_abs_start_error_ms << ",\n"
        << "    \"p90_abs_clause_region_start_error_ms\": " << grade_report.pause_alignment.clause_regions.p90_abs_start_error_ms << ",\n"
        << "    \"mean_abs_clause_region_end_error_ms\": " << grade_report.pause_alignment.clause_regions.mean_abs_end_error_ms << ",\n"
        << "    \"median_abs_clause_region_end_error_ms\": " << grade_report.pause_alignment.clause_regions.median_abs_end_error_ms << ",\n"
        << "    \"p90_abs_clause_region_end_error_ms\": " << grade_report.pause_alignment.clause_regions.p90_abs_end_error_ms << ",\n"
        << "    \"clause_region_count_mismatch\": " << (grade_report.pause_alignment.clause_regions.count_mismatch ? "true" : "false") << ",\n"
        << "    \"mean_clause_region_lead_in_ms\": " << grade_report.pause_alignment.clause_regions.mean_lead_in_ms << ",\n"
        << "    \"mean_clause_region_tail_out_ms\": " << grade_report.pause_alignment.clause_regions.mean_tail_out_ms << "\n"
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
};

static RunnerCliConfig parse_cli_config(int argc, char** argv)
{
    RunnerCliConfig cli;
    bool root_set = false;

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
        }
        else if (starts_with(arg, "--tick-ms="))
        {
            cli.stream.playback_tick_seconds = parse_ms(arg.substr(std::string("--tick-ms=").size()), "--tick-ms");
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
    advance_playback_to(std::max(0.0f, duration_seconds - buffer_seconds));

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
            session.BeginLine(begin);
            run_streamed_runtime_session(session, wav, stream);

            const auto& plan = session.GetTextPlan();
            const auto& speech = session.GetSpeechIslands();
            const auto& gap_candidates = session.GetSpeechDetector().GetGapCandidates();
            const auto& committed = session.GetCommittedTrack();

            const fs::path case_dir = out_root / stem;
            fs::create_directories(case_dir);
            write_text(case_dir / "planned.csv", planned_csv(plan));
            write_text(case_dir / "speech_regions.csv", speech_csv(speech));
            write_text(case_dir / "gap_candidates.csv", gap_candidates_csv(gap_candidates));
            write_text(case_dir / "occupancy_frames.csv", occupancy_frames_csv(session.GetSpeechDetector().GetFeatureFrames()));
            write_text(case_dir / "committed.csv", committed_csv(committed));
            write_text(case_dir / "commit_decisions.csv", commit_decisions_csv(committed));
            write_text(case_dir / "stream_tail.csv", stream_tail_csv(session.GetStreamTailDiagnosticRow()));
            GradeReport report;
            const fs::path gold_case_dir = root / "inputs" / "gold" / stem;
            const fs::path gold_visemes_path = gold_case_dir / "visemes.csv";
            const fs::path gold_words_path = gold_case_dir / "words.csv";
            const fs::path gold_speech_path = gold_case_dir / "speech.csv";
            const bool gold_available = fs::exists(gold_visemes_path) && fs::exists(gold_words_path) && fs::exists(gold_speech_path);
            size_t gold_viseme_count = 0;
            if (gold_available)
            {
                const auto handmade = read_handmade_csv(gold_visemes_path);
                const auto gold_words = read_gold_words_csv(gold_words_path);
                const auto gold_speech = read_gold_speech_csv(gold_speech_path);
                gold_viseme_count = handmade.size();
                report = grade(committed, speech, handmade, gold_words, gold_speech);
                StreamingDetectorReport detector_report;
                write_text(
                    case_dir / "streaming_detector_frames.csv",
                    streaming_detector_frames_csv(
                        detector_report,
                        session.GetSpeechDetector().GetFeatureFrames(),
                        gold_words,
                        gold_speech,
                        static_cast<double>(wav.samples.size()) / static_cast<double>(std::max(wav.sample_rate * wav.channels, 1))));
                write_text(case_dir / "streaming_detector_summary.json", streaming_detector_summary_json(detector_report));
                write_text(case_dir / "word_onset_diagnostics.csv", word_onset_diagnostics_csv(committed, handmade, gold_words));
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
