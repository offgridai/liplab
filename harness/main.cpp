#include "Lipsync/OffgridAILipsyncRuntimeAdapter.h"
#include "Lipsync/OffgridAIOnlinePhoneAligner.h"
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

struct AudioWordBoundaryCandidate
{
    double time = 0.0;
    double salience = 0.0;
    double energy_valley_score = 0.0;
    double trough_score = 0.0;
    double low_evidence_score = 0.0;
    double spectral_change_score = 0.0;
    double spectral_novelty_score = 0.0;
    double voicing_change_score = 0.0;
    double flux_recovery_score = 0.0;
    double reengagement_score = 0.0;
    double slope_score = 0.0;
    double island_adaptive_score = 0.0;
    double stable_voicing_penalty = 0.0;
    double lexical_boundary_score = 0.0;
    double false_positive_score = 0.0;
    int salience_rank = 0;
    int lexical_rank = 0;
    std::string diagnostic_label;
    double nearest_mfa_boundary_ms = 0.0;
    double nearest_mfa_error_ms = 0.0;
    int nearest_mfa_index = -1;
    bool selected = false;
};

struct AudioWordBoundaryEval
{
    int reference_count = 0;
    int candidate_count = 0;
    int selected_count = 0;
    double selected_precision_50 = 0.0;
    double selected_recall_50 = 0.0;
    double selected_f1_50 = 0.0;
    double selected_precision_100 = 0.0;
    double selected_recall_100 = 0.0;
    double selected_f1_100 = 0.0;
    double selected_precision_150 = 0.0;
    double selected_recall_150 = 0.0;
    double selected_f1_150 = 0.0;
    double reference_nearest_candidate_median_ms = 0.0;
    double reference_nearest_candidate_p90_ms = 0.0;
    double candidate_nearest_reference_median_ms = 0.0;
    double candidate_nearest_reference_p90_ms = 0.0;
    double raw_candidate_recall_50 = 0.0;
    double raw_candidate_recall_100 = 0.0;
    double raw_candidate_recall_150 = 0.0;
    double top1_recall_100 = 0.0;
    double top2_recall_100 = 0.0;
    double top3_recall_100 = 0.0;
    double top5_recall_100 = 0.0;
    double candidate_positive_mean_score = 0.0;
    double candidate_negative_mean_score = 0.0;
    int candidate_positive_count = 0;
    int candidate_negative_count = 0;
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

static bool time_inside_any_island(double time_sec, const TArray<FOffgridAIStreamingSpeechIsland>& islands, double margin_sec = 0.060)
{
    for (const auto& island : islands)
    {
        const double start = static_cast<double>(island.AudioBufferStartSec) + margin_sec;
        const double end = static_cast<double>(island.AudioBufferEndSec) - margin_sec;
        if (end > start && time_sec >= start && time_sec <= end)
        {
            return true;
        }
    }
    return islands.Num() == 0;
}

static double detector_evidence_for_frame(const FOffgridAIStreamingAudioFeatureFrame& frame)
{
    const double voiced = static_cast<double>(frame.Periodicity) * 0.42
        + static_cast<double>(frame.RMSNorm) * 0.24
        + static_cast<double>(frame.MidBandNorm) * 0.10;
    const double unvoiced = static_cast<double>(frame.Flux) * 0.26
        + static_cast<double>(frame.HighBandNorm) * 0.12
        + static_cast<double>(frame.SpectralCentroidNorm) * 0.10;
    return std::clamp(voiced + unvoiced, 0.0, 1.0);
}

static std::vector<double> gold_internal_word_boundary_times(const std::vector<GoldWordTiming>& words)
{
    std::vector<double> out;
    for (size_t i = 1; i < words.size(); ++i)
    {
        const GoldWordTiming& prev = words[i - 1];
        const GoldWordTiming& cur = words[i];
        if (cur.start < 0.0 || cur.end <= cur.start) continue;
        if (prev.sentence_index >= 0 && cur.sentence_index >= 0 && prev.sentence_index != cur.sentence_index)
        {
            continue;
        }
        out.push_back(cur.start);
    }
    return out;
}

static double nearest_error_ms(double time_sec, const std::vector<double>& refs, int* out_index = nullptr)
{
    double best = std::numeric_limits<double>::infinity();
    int best_index = -1;
    for (size_t i = 0; i < refs.size(); ++i)
    {
        const double err = std::abs(time_sec - refs[i]) * 1000.0;
        if (err < best)
        {
            best = err;
            best_index = static_cast<int>(i);
        }
    }
    if (out_index) *out_index = best_index;
    return std::isfinite(best) ? best : 0.0;
}


static double lexical_boundary_classifier_score(const AudioWordBoundaryCandidate& c)
{
    // Lightweight supervised-classifier proxy. The features and coefficients are intentionally
    // explicit and dependency-free so the CSV can later be used to fit/replace them. True lexical
    // boundaries tend to have multiple independent cues: a short trough or valley, spectral novelty,
    // re-engagement, and a clean local peak. Intra-word false positives often have only one cue and
    // stable voiced energy on both sides.
    const bool multi_cue =
        ((c.energy_valley_score > 0.22) ? 1 : 0) +
        ((c.spectral_novelty_score > 0.20) ? 1 : 0) +
        ((c.reengagement_score > 0.20) ? 1 : 0) +
        ((c.voicing_change_score > 0.22) ? 1 : 0) +
        ((c.slope_score > 0.18) ? 1 : 0) >= 2;

    double score = 0.0;
    score += 0.30 * c.salience;
    score += 0.18 * c.spectral_novelty_score;
    score += 0.14 * c.energy_valley_score;
    score += 0.11 * c.reengagement_score;
    score += 0.09 * c.voicing_change_score;
    score += 0.08 * c.slope_score;
    score += 0.07 * c.trough_score;
    score += 0.05 * c.island_adaptive_score;
    score += multi_cue ? 0.08 : -0.04;
    score -= 0.16 * c.stable_voicing_penalty;
    // Very high flux without a valley/novelty combination is often a plosive/fricative event inside
    // a word rather than a word boundary.
    const double burst_only = std::clamp(c.flux_recovery_score - std::max(c.energy_valley_score, c.spectral_novelty_score), 0.0, 1.0);
    score -= 0.08 * burst_only;
    return std::clamp(score, 0.0, 1.0);
}


static std::string audio_boundary_reason_codes(const AudioWordBoundaryCandidate& c)
{
    std::vector<std::string> reasons;
    if (c.energy_valley_score >= 0.28) reasons.push_back("energy_valley");
    if (c.trough_score >= 0.55) reasons.push_back("short_trough");
    if (c.low_evidence_score >= 0.55) reasons.push_back("low_evidence");
    if (c.spectral_change_score >= 0.25) reasons.push_back("spectral_change");
    if (c.spectral_novelty_score >= 0.25) reasons.push_back("spectral_novelty");
    if (c.voicing_change_score >= 0.28) reasons.push_back("voicing_change");
    if (c.flux_recovery_score >= 0.28) reasons.push_back("flux_recovery");
    if (c.reengagement_score >= 0.28) reasons.push_back("reengagement");
    if (c.slope_score >= 0.25) reasons.push_back("slope_change");
    if (c.island_adaptive_score >= 0.25) reasons.push_back("island_adaptive_peak");
    if (c.stable_voicing_penalty >= 0.32) reasons.push_back("stable_voicing_penalty");
    if (reasons.empty()) reasons.push_back("weak_salience_only");
    std::ostringstream out;
    for (size_t i = 0; i < reasons.size(); ++i)
    {
        if (i > 0) out << '|';
        out << reasons[i];
    }
    return out.str();
}

static std::string audio_boundary_diagnostic_label(const AudioWordBoundaryCandidate& c)
{
    if (c.nearest_mfa_error_ms <= 50.0) return "mfa_boundary_50";
    if (c.nearest_mfa_error_ms <= 100.0) return "mfa_boundary_100";
    if (c.nearest_mfa_error_ms <= 150.0) return "mfa_boundary_150";
    if (c.low_evidence_score > 0.70 && c.trough_score > 0.55) return "pause_or_phrase_event";
    if (c.flux_recovery_score > 0.55 && c.energy_valley_score < 0.18 && c.spectral_novelty_score < 0.20) return "burst_or_release";
    if (c.stable_voicing_penalty > 0.35) return "stable_voiced_internal";
    if (c.spectral_novelty_score > 0.35 || c.voicing_change_score > 0.35) return "spectral_or_voicing_change";
    return "unclassified_false_candidate";
}

static std::vector<AudioWordBoundaryCandidate> detect_audio_word_boundary_candidates(
    const TArray<FOffgridAIStreamingAudioFeatureFrame>& frames,
    const TArray<FOffgridAIStreamingSpeechIsland>& islands,
    const std::vector<double>& gold_boundaries)
{
    std::vector<AudioWordBoundaryCandidate> raw;
    const int32 n = frames.Num();
    if (n < 13) return raw;

    auto avg = [&](int32 lo, int32 hi, auto getter) -> double
    {
        lo = std::max(0, lo);
        hi = std::min(n - 1, hi);
        if (hi < lo) return 0.0;
        double sum = 0.0;
        int count = 0;
        for (int32 i = lo; i <= hi; ++i)
        {
            sum += getter(frames[i]);
            ++count;
        }
        return count > 0 ? sum / static_cast<double>(count) : 0.0;
    };

    auto maxv = [&](int32 lo, int32 hi, auto getter) -> double
    {
        lo = std::max(0, lo);
        hi = std::min(n - 1, hi);
        double best = 0.0;
        for (int32 i = lo; i <= hi; ++i) best = std::max(best, getter(frames[i]));
        return best;
    };

    auto minv = [&](int32 lo, int32 hi, auto getter) -> double
    {
        lo = std::max(0, lo);
        hi = std::min(n - 1, hi);
        double best = 1.0;
        for (int32 i = lo; i <= hi; ++i) best = std::min(best, getter(frames[i]));
        return best;
    };

    auto band_dist = [](double a0, double a1, double a2, double a3, double b0, double b1, double b2, double b3) -> double
    {
        const double dot = a0*b0 + a1*b1 + a2*b2 + a3*b3;
        const double na = std::sqrt(a0*a0 + a1*a1 + a2*a2 + a3*a3);
        const double nb = std::sqrt(b0*b0 + b1*b1 + b2*b2 + b3*b3);
        const double cosine_distance = (na > 1e-6 && nb > 1e-6) ? (1.0 - dot / (na * nb)) : 0.0;
        const double l1 = std::abs(a0-b0) + std::abs(a1-b1) + std::abs(a2-b2) + std::abs(a3-b3);
        return std::clamp(0.60 * cosine_distance + 0.45 * l1, 0.0, 1.0);
    };

    std::vector<double> salience(static_cast<size_t>(n), 0.0);
    std::vector<double> base_salience(static_cast<size_t>(n), 0.0);
    std::vector<AudioWordBoundaryCandidate> per_frame(static_cast<size_t>(n));

    for (int32 i = 5; i + 5 < n; ++i)
    {
        const auto& f = frames[i];
        const double t = static_cast<double>(f.AudioBufferCenterSec);
        if (!time_inside_any_island(t, islands, 0.045)) continue;

        // Multiscale context windows. Word boundaries may be short closures or spectral changes
        // with little silence, so use both near and wider windows.
        const double pre_rms_near = avg(i - 4, i - 1, [](const auto& x) { return static_cast<double>(x.RMSNorm); });
        const double post_rms_near = avg(i + 1, i + 4, [](const auto& x) { return static_cast<double>(x.RMSNorm); });
        const double pre_rms_wide = maxv(i - 8, i - 3, [](const auto& x) { return static_cast<double>(x.RMSNorm); });
        const double post_rms_wide = maxv(i + 3, i + 8, [](const auto& x) { return static_cast<double>(x.RMSNorm); });
        const double local_rms = minv(i - 1, i + 1, [](const auto& x) { return static_cast<double>(x.RMSNorm); });
        const double context_rms = std::max(0.05, 0.5 * (std::max(pre_rms_near, pre_rms_wide) + std::max(post_rms_near, post_rms_wide)));
        const double valley_depth = std::max(0.0, (context_rms - local_rms) / context_rms);
        const double energy_valley = std::clamp(valley_depth, 0.0, 1.0);

        const double evidence = detector_evidence_for_frame(f);
        const double pre_evidence = avg(i - 5, i - 2, [](const auto& x) { return detector_evidence_for_frame(x); });
        const double post_evidence = avg(i + 2, i + 5, [](const auto& x) { return detector_evidence_for_frame(x); });
        const double low_evidence = std::clamp(1.0 - evidence / 0.22, 0.0, 1.0);
        const double context_evidence = std::min(pre_evidence, post_evidence);
        const double reengagement = std::clamp(context_evidence * low_evidence * 2.7, 0.0, 1.0);

        int trough_frames = 1;
        for (int32 j = i - 1; j >= 0; --j)
        {
            if (detector_evidence_for_frame(frames[j]) > 0.15 && frames[j].RMSNorm > 0.19f) break;
            ++trough_frames;
            if (trough_frames >= 14) break;
        }
        for (int32 j = i + 1; j < n; ++j)
        {
            if (detector_evidence_for_frame(frames[j]) > 0.15 && frames[j].RMSNorm > 0.19f) break;
            ++trough_frames;
            if (trough_frames >= 14) break;
        }
        const double trough_ms = trough_frames * 10.0;
        // Lexical boundaries are usually short: closures/brief troughs help, very long troughs are more phrase/sentence-like.
        const double trough_score = std::clamp(1.0 - std::abs(trough_ms - 55.0) / 95.0, 0.0, 1.0);

        const double pre_low = avg(i - 5, i - 2, [](const auto& x) { return static_cast<double>(x.LowBandNorm); });
        const double post_low = avg(i + 2, i + 5, [](const auto& x) { return static_cast<double>(x.LowBandNorm); });
        const double pre_mid = avg(i - 5, i - 2, [](const auto& x) { return static_cast<double>(x.MidBandNorm); });
        const double post_mid = avg(i + 2, i + 5, [](const auto& x) { return static_cast<double>(x.MidBandNorm); });
        const double pre_high = avg(i - 5, i - 2, [](const auto& x) { return static_cast<double>(x.HighBandNorm); });
        const double post_high = avg(i + 2, i + 5, [](const auto& x) { return static_cast<double>(x.HighBandNorm); });
        const double pre_cent = avg(i - 5, i - 2, [](const auto& x) { return static_cast<double>(x.SpectralCentroidNorm); });
        const double post_cent = avg(i + 2, i + 5, [](const auto& x) { return static_cast<double>(x.SpectralCentroidNorm); });
        const double spectral_change = std::clamp(
            std::abs(pre_low - post_low) * 0.85 + std::abs(pre_mid - post_mid) * 0.95 + std::abs(pre_high - post_high) * 0.95 + std::abs(pre_cent - post_cent) * 2.1,
            0.0,
            1.0);

        // Spectral novelty: compare the band vector before and after the candidate. This catches
        // word transitions with continuous energy but a sharp phonetic/spectral change.
        const double novelty_near = band_dist(pre_low, pre_mid, pre_high, pre_cent, post_low, post_mid, post_high, post_cent);
        const double pre_low_w = avg(i - 9, i - 4, [](const auto& x) { return static_cast<double>(x.LowBandNorm); });
        const double post_low_w = avg(i + 4, i + 9, [](const auto& x) { return static_cast<double>(x.LowBandNorm); });
        const double pre_mid_w = avg(i - 9, i - 4, [](const auto& x) { return static_cast<double>(x.MidBandNorm); });
        const double post_mid_w = avg(i + 4, i + 9, [](const auto& x) { return static_cast<double>(x.MidBandNorm); });
        const double pre_high_w = avg(i - 9, i - 4, [](const auto& x) { return static_cast<double>(x.HighBandNorm); });
        const double post_high_w = avg(i + 4, i + 9, [](const auto& x) { return static_cast<double>(x.HighBandNorm); });
        const double pre_cent_w = avg(i - 9, i - 4, [](const auto& x) { return static_cast<double>(x.SpectralCentroidNorm); });
        const double post_cent_w = avg(i + 4, i + 9, [](const auto& x) { return static_cast<double>(x.SpectralCentroidNorm); });
        const double novelty_wide = band_dist(pre_low_w, pre_mid_w, pre_high_w, pre_cent_w, post_low_w, post_mid_w, post_high_w, post_cent_w);
        const double spectral_novelty = std::clamp(0.60 * novelty_near + 0.40 * novelty_wide, 0.0, 1.0);

        const double pre_voicing = avg(i - 5, i - 2, [](const auto& x) { return static_cast<double>(x.Periodicity); });
        const double post_voicing = avg(i + 2, i + 5, [](const auto& x) { return static_cast<double>(x.Periodicity); });
        const double voicing_change = std::clamp(std::abs(pre_voicing - post_voicing) * 1.8, 0.0, 1.0);
        const double flux_recovery = std::clamp(maxv(i + 1, i + 6, [](const auto& x) { return static_cast<double>(x.Flux); }) * 4.3, 0.0, 1.0);
        const double pre_slope = avg(i - 2, i - 1, [](const auto& x) { return static_cast<double>(x.RMSNorm); }) - avg(i - 5, i - 4, [](const auto& x) { return static_cast<double>(x.RMSNorm); });
        const double post_slope = avg(i + 4, i + 5, [](const auto& x) { return static_cast<double>(x.RMSNorm); }) - avg(i + 1, i + 2, [](const auto& x) { return static_cast<double>(x.RMSNorm); });
        const double slope_score = std::clamp((-pre_slope * 4.0) + (post_slope * 4.0), 0.0, 1.0);
        const double stable_voicing_penalty = std::clamp(std::min(pre_voicing, post_voicing) * std::min(pre_rms_near, post_rms_near) * (1.0 - std::max(spectral_change, spectral_novelty)), 0.0, 1.0);

        AudioWordBoundaryCandidate c;
        c.time = t;
        c.energy_valley_score = energy_valley;
        c.trough_score = trough_score;
        c.low_evidence_score = low_evidence;
        c.spectral_change_score = spectral_change;
        c.spectral_novelty_score = spectral_novelty;
        c.voicing_change_score = voicing_change;
        c.flux_recovery_score = flux_recovery;
        c.reengagement_score = reengagement;
        c.slope_score = slope_score;
        c.stable_voicing_penalty = stable_voicing_penalty;
        c.salience = std::clamp(
            0.18 * energy_valley
            + 0.13 * low_evidence
            + 0.09 * trough_score
            + 0.16 * spectral_change
            + 0.18 * spectral_novelty
            + 0.08 * voicing_change
            + 0.07 * flux_recovery
            + 0.13 * reengagement
            + 0.06 * slope_score
            - 0.22 * stable_voicing_penalty,
            0.0,
            1.0);
        base_salience[static_cast<size_t>(i)] = c.salience;
        per_frame[static_cast<size_t>(i)] = c;
    }

    // Per-island robust normalization. This makes the detector less sensitive to absolute speaker volume
    // and line loudness, and follows the novelty-detection practice of picking peaks above a local floor.
    for (int32 island_idx = 0; island_idx < islands.Num() || (islands.Num() == 0 && island_idx == 0); ++island_idx)
    {
        const double island_start = islands.Num() > 0 ? static_cast<double>(islands[island_idx].AudioBufferStartSec) : static_cast<double>(frames[0].AudioBufferCenterSec);
        const double island_end = islands.Num() > 0 ? static_cast<double>(islands[island_idx].AudioBufferEndSec) : static_cast<double>(frames[n - 1].AudioBufferCenterSec);
        std::vector<double> values;
        for (int32 i = 0; i < n; ++i)
        {
            const double t = static_cast<double>(frames[i].AudioBufferCenterSec);
            if (t < island_start + 0.045 || t > island_end - 0.045) continue;
            if (base_salience[static_cast<size_t>(i)] > 0.0) values.push_back(base_salience[static_cast<size_t>(i)]);
        }
        if (values.empty()) continue;
        const double med = percentile_ms(values, 0.50);
        std::vector<double> deviations;
        deviations.reserve(values.size());
        for (double v : values) deviations.push_back(std::abs(v - med));
        const double mad = std::max(0.025, percentile_ms(deviations, 0.50));
        const double p85 = percentile_ms(values, 0.85);

        for (int32 i = 0; i < n; ++i)
        {
            const double t = static_cast<double>(frames[i].AudioBufferCenterSec);
            if (t < island_start + 0.045 || t > island_end - 0.045) continue;
            if (base_salience[static_cast<size_t>(i)] <= 0.0) continue;
            AudioWordBoundaryCandidate c = per_frame[static_cast<size_t>(i)];
            const double robust_z01 = std::clamp((c.salience - med) / (3.0 * mad), 0.0, 1.0);
            const double percentile_boost = c.salience >= p85 ? 0.10 : 0.0;
            c.island_adaptive_score = robust_z01;
            c.salience = std::clamp(0.78 * c.salience + 0.22 * robust_z01 + percentile_boost, 0.0, 1.0);
            salience[static_cast<size_t>(i)] = c.salience;
            per_frame[static_cast<size_t>(i)] = c;
        }
    }

    std::vector<AudioWordBoundaryCandidate> local_maxima;
    for (int32 frame_index = 5; frame_index + 5 < n; ++frame_index)
    {
        AudioWordBoundaryCandidate c = per_frame[static_cast<size_t>(frame_index)];
        if (c.salience <= 0.0) continue;
        bool is_peak = true;
        for (int32 j = std::max(0, frame_index - 4); j <= std::min(n - 1, frame_index + 4); ++j)
        {
            if (j != frame_index && salience[static_cast<size_t>(j)] > c.salience)
            {
                is_peak = false;
                break;
            }
        }
        if (is_peak && c.salience >= 0.31)
        {
            int nearest = -1;
            c.nearest_mfa_error_ms = nearest_error_ms(c.time, gold_boundaries, &nearest);
            c.nearest_mfa_index = nearest;
            c.nearest_mfa_boundary_ms = nearest >= 0 ? gold_boundaries[static_cast<size_t>(nearest)] * 1000.0 : 0.0;
            local_maxima.push_back(c);
        }
    }

    // Stage 1: local peak thinning. Keep a salience field, but avoid returning clusters of
    // adjacent local maxima around the same acoustic event. This is deliberately looser than
    // final selection: it preserves a candidate set for the sequence optimizer below.
    std::sort(local_maxima.begin(), local_maxima.end(), [](const auto& a, const auto& b) { return a.salience > b.salience; });
    std::vector<AudioWordBoundaryCandidate> peak_candidates_by_score;
    for (auto c : local_maxima)
    {
        bool too_close = false;
        for (const auto& kept : peak_candidates_by_score)
        {
            if (std::abs(kept.time - c.time) < 0.075)
            {
                too_close = true;
                break;
            }
        }
        if (too_close) continue;
        const bool complementary = (c.energy_valley_score > 0.28 && c.spectral_novelty_score > 0.22) ||
                                   (c.reengagement_score > 0.28 && c.spectral_change_score > 0.18) ||
                                   (c.voicing_change_score > 0.30 && c.spectral_novelty_score > 0.18);
        if (c.salience >= 0.32 || complementary)
        {
            c.selected = false;
            peak_candidates_by_score.push_back(c);
        }
    }
    std::sort(peak_candidates_by_score.begin(), peak_candidates_by_score.end(), [](const auto& a, const auto& b) { return a.time < b.time; });

    auto feature_reward = [](const AudioWordBoundaryCandidate& c) -> double
    {
        const bool complementary = (c.energy_valley_score > 0.28 && c.spectral_novelty_score > 0.22) ||
                                   (c.reengagement_score > 0.28 && c.spectral_change_score > 0.18) ||
                                   (c.voicing_change_score > 0.30 && c.spectral_novelty_score > 0.18);
        // Selection reward deliberately has a fixed cost. Greedy local maxima tended to
        // over-select acoustically interesting intra-word events. A candidate now needs either
        // strong salience or multiple independent cues to survive the global optimizer.
        double r = 1.35 * c.salience - 0.62;
        r += complementary ? 0.10 : 0.0;
        r += 0.08 * c.island_adaptive_score;
        r += 0.05 * c.trough_score;
        r -= 0.18 * c.stable_voicing_penalty;
        return r;
    };

    auto gap_score = [](double gap_sec) -> double
    {
        // Audio-only lexical-boundary candidates should usually be separated by at least a
        // short syllabic interval. Very small gaps are duplicate peaks. Moderate gaps are
        // plausible word spacing. Long gaps are allowed but not especially rewarded.
        const double gap_ms = gap_sec * 1000.0;
        if (gap_ms < 95.0) return -2.00;
        if (gap_ms < 135.0) return -0.55;
        if (gap_ms < 210.0) return 0.02;
        if (gap_ms < 620.0) return 0.09;
        if (gap_ms < 950.0) return 0.01;
        return -0.10;
    };

    std::vector<AudioWordBoundaryCandidate> sequence_selected;

    // Stage 2: per-island sequence optimization. This borrows from onset/beat tracking and
    // change-point decoding: select a coherent subset of peaks rather than accepting every
    // local maximum independently. It remains audio-only; no transcript or MFA count is used.
    for (int32 island_idx = 0; island_idx < islands.Num() || (islands.Num() == 0 && island_idx == 0); ++island_idx)
    {
        const double island_start = islands.Num() > 0 ? static_cast<double>(islands[island_idx].AudioBufferStartSec) : static_cast<double>(frames[0].AudioBufferCenterSec);
        const double island_end = islands.Num() > 0 ? static_cast<double>(islands[island_idx].AudioBufferEndSec) : static_cast<double>(frames[n - 1].AudioBufferCenterSec);
        const double island_duration = std::max(0.0, island_end - island_start);

        std::vector<AudioWordBoundaryCandidate> peaks;
        for (const auto& c : peak_candidates_by_score)
        {
            if (c.time >= island_start + 0.055 && c.time <= island_end - 0.055)
            {
                peaks.push_back(c);
            }
        }
        if (peaks.empty()) continue;

        const int m = static_cast<int>(peaks.size());
        std::vector<double> dp(static_cast<size_t>(m), -1e9);
        std::vector<int> prev(static_cast<size_t>(m), -1);
        for (int i = 0; i < m; ++i)
        {
            const double edge_penalty = (peaks[i].time - island_start < 0.11 || island_end - peaks[i].time < 0.11) ? -0.12 : 0.0;
            dp[static_cast<size_t>(i)] = feature_reward(peaks[i]) + edge_penalty;
            for (int j = 0; j < i; ++j)
            {
                const double gap = peaks[i].time - peaks[j].time;
                if (gap < 0.095) continue;
                const double score = dp[static_cast<size_t>(j)] + feature_reward(peaks[i]) + gap_score(gap);
                if (score > dp[static_cast<size_t>(i)])
                {
                    dp[static_cast<size_t>(i)] = score;
                    prev[static_cast<size_t>(i)] = j;
                }
            }
        }

        int best_i = -1;
        double best_score = 0.0; // allow selecting no boundaries in very short/flat islands
        for (int i = 0; i < m; ++i)
        {
            if (dp[static_cast<size_t>(i)] > best_score)
            {
                best_score = dp[static_cast<size_t>(i)];
                best_i = i;
            }
        }

        std::vector<int> path;
        while (best_i >= 0)
        {
            path.push_back(best_i);
            best_i = prev[static_cast<size_t>(best_i)];
        }
        std::reverse(path.begin(), path.end());

        // Density guard. The optimizer is still audio-only, but spoken lexical/prosodic
        // transitions have a plausible upper density. Keep the strongest path members if a
        // noisy island tries to emit far more candidates than a human speaking rate permits.
        const int soft_cap = std::max(1, static_cast<int>(std::ceil(island_duration * 4.2)) + 1);
        if (static_cast<int>(path.size()) > soft_cap)
        {
            std::sort(path.begin(), path.end(), [&](int a, int b) {
                return feature_reward(peaks[static_cast<size_t>(a)]) > feature_reward(peaks[static_cast<size_t>(b)]);
            });
            path.resize(static_cast<size_t>(soft_cap));
            std::sort(path.begin(), path.end(), [&](int a, int b) { return peaks[static_cast<size_t>(a)].time < peaks[static_cast<size_t>(b)].time; });
        }

        for (int idx : path)
        {
            auto c = peaks[static_cast<size_t>(idx)];
            c.selected = true;
            sequence_selected.push_back(c);
        }
    }

    // Return the full candidate stream, with only the sequence-optimized subset marked selected.
    // This keeps diagnostics useful: CSV rows show rejected peaks as well as final observations.
    std::vector<AudioWordBoundaryCandidate> result = peak_candidates_by_score;
    for (auto& c : result)
    {
        c.selected = false;
        for (const auto& kept : sequence_selected)
        {
            if (std::abs(kept.time - c.time) < 0.001)
            {
                c.selected = true;
                break;
            }
        }
        c.lexical_boundary_score = lexical_boundary_classifier_score(c);
        c.false_positive_score = std::clamp(1.0 - c.lexical_boundary_score + 0.20 * c.stable_voicing_penalty, 0.0, 1.0);
        c.diagnostic_label = audio_boundary_diagnostic_label(c);
    }

    std::vector<size_t> by_salience(result.size());
    std::iota(by_salience.begin(), by_salience.end(), static_cast<size_t>(0));
    std::sort(by_salience.begin(), by_salience.end(), [&](size_t a, size_t b) { return result[a].salience > result[b].salience; });
    for (size_t rank = 0; rank < by_salience.size(); ++rank) result[by_salience[rank]].salience_rank = static_cast<int>(rank + 1);

    std::vector<size_t> by_lexical(result.size());
    std::iota(by_lexical.begin(), by_lexical.end(), static_cast<size_t>(0));
    std::sort(by_lexical.begin(), by_lexical.end(), [&](size_t a, size_t b) { return result[a].lexical_boundary_score > result[b].lexical_boundary_score; });
    for (size_t rank = 0; rank < by_lexical.size(); ++rank) result[by_lexical[rank]].lexical_rank = static_cast<int>(rank + 1);

    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) { return a.time < b.time; });
    return result;
}


static AudioWordBoundaryEval evaluate_audio_word_boundary_candidates(
    const std::vector<AudioWordBoundaryCandidate>& candidates,
    const std::vector<double>& gold_boundaries)
{
    AudioWordBoundaryEval eval;
    eval.reference_count = static_cast<int>(gold_boundaries.size());
    eval.candidate_count = static_cast<int>(candidates.size());
    for (const auto& c : candidates) if (c.selected) ++eval.selected_count;

    std::vector<double> candidate_errors;
    std::vector<double> selected_errors;
    double pos_score_sum = 0.0;
    double neg_score_sum = 0.0;
    for (const auto& c : candidates)
    {
        candidate_errors.push_back(c.nearest_mfa_error_ms);
        if (c.selected) selected_errors.push_back(c.nearest_mfa_error_ms);
        if (c.nearest_mfa_error_ms <= 100.0)
        {
            ++eval.candidate_positive_count;
            pos_score_sum += c.lexical_boundary_score;
        }
        else
        {
            ++eval.candidate_negative_count;
            neg_score_sum += c.lexical_boundary_score;
        }
    }
    eval.candidate_positive_mean_score = eval.candidate_positive_count > 0 ? pos_score_sum / static_cast<double>(eval.candidate_positive_count) : 0.0;
    eval.candidate_negative_mean_score = eval.candidate_negative_count > 0 ? neg_score_sum / static_cast<double>(eval.candidate_negative_count) : 0.0;

    std::vector<double> reference_errors;
    for (double ref : gold_boundaries)
    {
        double best = std::numeric_limits<double>::infinity();
        for (const auto& c : candidates)
        {
            if (!c.selected) continue;
            best = std::min(best, std::abs(c.time - ref) * 1000.0);
        }
        if (std::isfinite(best)) reference_errors.push_back(best);
    }

    auto pr_at = [&](double threshold_ms, double& precision, double& recall, double& f1)
    {
        int selected_hits = 0;
        int ref_hits = 0;
        for (double err : selected_errors) if (err <= threshold_ms) ++selected_hits;
        for (double err : reference_errors) if (err <= threshold_ms) ++ref_hits;
        precision = eval.selected_count > 0 ? static_cast<double>(selected_hits) / static_cast<double>(eval.selected_count) : 0.0;
        recall = eval.reference_count > 0 ? static_cast<double>(ref_hits) / static_cast<double>(eval.reference_count) : 0.0;
        f1 = (precision > 0.0 || recall > 0.0) ? (2.0 * precision * recall / (precision + recall)) : 0.0;
    };
    pr_at(50.0, eval.selected_precision_50, eval.selected_recall_50, eval.selected_f1_50);
    pr_at(100.0, eval.selected_precision_100, eval.selected_recall_100, eval.selected_f1_100);
    pr_at(150.0, eval.selected_precision_150, eval.selected_recall_150, eval.selected_f1_150);
    eval.reference_nearest_candidate_median_ms = percentile_ms(reference_errors, 0.50);
    eval.reference_nearest_candidate_p90_ms = percentile_ms(reference_errors, 0.90);
    eval.candidate_nearest_reference_median_ms = percentile_ms(candidate_errors, 0.50);
    eval.candidate_nearest_reference_p90_ms = percentile_ms(candidate_errors, 0.90);
    auto raw_recall_at = [&](double threshold_ms) -> double
    {
        if (eval.reference_count <= 0) return 0.0;
        int hits = 0;
        for (double ref : gold_boundaries)
        {
            bool found = false;
            for (const auto& c : candidates)
            {
                if (std::abs(c.time - ref) * 1000.0 <= threshold_ms)
                {
                    found = true;
                    break;
                }
            }
            if (found) ++hits;
        }
        return static_cast<double>(hits) / static_cast<double>(eval.reference_count);
    };
    eval.raw_candidate_recall_50 = raw_recall_at(50.0);
    eval.raw_candidate_recall_100 = raw_recall_at(100.0);
    eval.raw_candidate_recall_150 = raw_recall_at(150.0);

    auto topk_recall_at_100 = [&](int k) -> double
    {
        if (eval.reference_count <= 0) return 0.0;
        int hits = 0;
        for (double ref : gold_boundaries)
        {
            std::vector<const AudioWordBoundaryCandidate*> nearby;
            for (const auto& c : candidates)
            {
                if (std::abs(c.time - ref) * 1000.0 <= 250.0) nearby.push_back(&c);
            }
            std::sort(nearby.begin(), nearby.end(), [](const auto* a, const auto* b) { return a->lexical_boundary_score > b->lexical_boundary_score; });
            const int limit = std::min(k, static_cast<int>(nearby.size()));
            bool found = false;
            for (int i = 0; i < limit; ++i)
            {
                if (std::abs(nearby[static_cast<size_t>(i)]->time - ref) * 1000.0 <= 100.0)
                {
                    found = true;
                    break;
                }
            }
            if (found) ++hits;
        }
        return static_cast<double>(hits) / static_cast<double>(eval.reference_count);
    };
    eval.top1_recall_100 = topk_recall_at_100(1);
    eval.top2_recall_100 = topk_recall_at_100(2);
    eval.top3_recall_100 = topk_recall_at_100(3);
    eval.top5_recall_100 = topk_recall_at_100(5);

    return eval;
}

static std::string audio_word_boundary_candidates_csv(const std::vector<AudioWordBoundaryCandidate>& candidates)
{
    std::ostringstream out;
    out << "candidate_index,time_ms,salience,lexical_boundary_score,false_positive_score,salience_rank,lexical_rank,diagnostic_label,reason_codes,selected,energy_valley_score,trough_score,low_evidence_score,spectral_change_score,spectral_novelty_score,voicing_change_score,flux_recovery_score,reengagement_score,slope_score,island_adaptive_score,stable_voicing_penalty,nearest_mfa_boundary_ms,nearest_mfa_error_ms,nearest_mfa_index,within_50ms,within_100ms,within_150ms\n";
    out << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < candidates.size(); ++i)
    {
        const auto& c = candidates[i];
        out << i << ','
            << c.time * 1000.0 << ','
            << c.salience << ','
            << c.lexical_boundary_score << ','
            << c.false_positive_score << ','
            << c.salience_rank << ','
            << c.lexical_rank << ','
            << c.diagnostic_label << ','
            << audio_boundary_reason_codes(c) << ','
            << (c.selected ? 1 : 0) << ','
            << c.energy_valley_score << ','
            << c.trough_score << ','
            << c.low_evidence_score << ','
            << c.spectral_change_score << ','
            << c.spectral_novelty_score << ','
            << c.voicing_change_score << ','
            << c.flux_recovery_score << ','
            << c.reengagement_score << ','
            << c.slope_score << ','
            << c.island_adaptive_score << ','
            << c.stable_voicing_penalty << ','
            << c.nearest_mfa_boundary_ms << ','
            << c.nearest_mfa_error_ms << ','
            << c.nearest_mfa_index << ','
            << (c.nearest_mfa_error_ms <= 50.0 ? 1 : 0) << ','
            << (c.nearest_mfa_error_ms <= 100.0 ? 1 : 0) << ','
            << (c.nearest_mfa_error_ms <= 150.0 ? 1 : 0) << '\n';
    }
    return out.str();
}

static std::string audio_word_boundary_grade_json(const AudioWordBoundaryEval& eval)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(6);
    out << "{\n"
        << "  \"reference_count\": " << eval.reference_count << ",\n"
        << "  \"candidate_count\": " << eval.candidate_count << ",\n"
        << "  \"selected_count\": " << eval.selected_count << ",\n"
        << "  \"precision_50\": " << eval.selected_precision_50 << ",\n"
        << "  \"recall_50\": " << eval.selected_recall_50 << ",\n"
        << "  \"f1_50\": " << eval.selected_f1_50 << ",\n"
        << "  \"precision_100\": " << eval.selected_precision_100 << ",\n"
        << "  \"recall_100\": " << eval.selected_recall_100 << ",\n"
        << "  \"f1_100\": " << eval.selected_f1_100 << ",\n"
        << "  \"precision_150\": " << eval.selected_precision_150 << ",\n"
        << "  \"recall_150\": " << eval.selected_recall_150 << ",\n"
        << "  \"f1_150\": " << eval.selected_f1_150 << ",\n"
        << "  \"reference_nearest_candidate_median_ms\": " << eval.reference_nearest_candidate_median_ms << ",\n"
        << "  \"reference_nearest_candidate_p90_ms\": " << eval.reference_nearest_candidate_p90_ms << ",\n"
        << "  \"candidate_nearest_reference_median_ms\": " << eval.candidate_nearest_reference_median_ms << ",\n"
        << "  \"candidate_nearest_reference_p90_ms\": " << eval.candidate_nearest_reference_p90_ms << ",\n"
        << "  \"raw_candidate_recall_50\": " << eval.raw_candidate_recall_50 << ",\n"
        << "  \"raw_candidate_recall_100\": " << eval.raw_candidate_recall_100 << ",\n"
        << "  \"raw_candidate_recall_150\": " << eval.raw_candidate_recall_150 << ",\n"
        << "  \"top1_recall_100\": " << eval.top1_recall_100 << ",\n"
        << "  \"top2_recall_100\": " << eval.top2_recall_100 << ",\n"
        << "  \"top3_recall_100\": " << eval.top3_recall_100 << ",\n"
        << "  \"top5_recall_100\": " << eval.top5_recall_100 << ",\n"
        << "  \"candidate_positive_count\": " << eval.candidate_positive_count << ",\n"
        << "  \"candidate_negative_count\": " << eval.candidate_negative_count << ",\n"
        << "  \"candidate_positive_mean_score\": " << eval.candidate_positive_mean_score << ",\n"
        << "  \"candidate_negative_mean_score\": " << eval.candidate_negative_mean_score << "\n"
        << "}\n";
    return out.str();
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
    out << "index,start,end,last_speech,started,ended\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& island : islands)
    {
        out << island.IslandIndex << ','
            << island.AudioBufferStartSec << ','
            << island.AudioBufferEndSec << ','
            << island.AudioBufferLastSpeechSec << ','
            << (island.bStarted ? 1 : 0) << ','
            << (island.bEnded ? 1 : 0) << '\n';
    }
    return out.str();
}

static std::string committed_csv(const FOffgridAIAlignedVisemeTrack& track)
{
    std::ostringstream out;
    out << "index,start,center,end,pose,word,word_index,phrase_index,sentence_index,strength,reason,alignment_reason,source_phone_index,source_phone_base,source_phone_class,aligned_phone_start,aligned_phone_end,alignment_confidence,alignment_score_gap,alignment_observed_duration,alignment_expected_duration,commit_playback,commit_lead,commit_confidence_threshold,commit_alignable_lag,stable_by_confidence,stable_by_boundary,stable_by_duration,stable_by_lead,stable_by_lag,mapped_to_observed_speech\n";
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
            << to_std(event.AlignmentReason) << ','
            << event.SourcePhoneIndex << ','
            << to_std(event.SourcePhoneBase) << ','
            << to_std(event.SourcePhoneClass) << ','
            << event.AlignedPhoneStartSeconds << ','
            << event.AlignedPhoneEndSeconds << ','
            << event.AlignmentConfidence << ','
            << event.AlignmentScoreGap << ','
            << event.AlignmentObservedDurationSeconds << ','
            << event.AlignmentExpectedDurationSeconds << ','
            << event.CommitPlaybackSeconds << ','
            << event.CommitLeadSeconds << ','
            << event.CommitConfidenceThreshold << ','
            << event.CommitAlignableLagSeconds << ','
            << (event.bCommitStableByConfidence ? 1 : 0) << ','
            << (event.bCommitStableByBoundary ? 1 : 0) << ','
            << (event.bCommitStableByDuration ? 1 : 0) << ','
            << (event.bCommitStableByLead ? 1 : 0) << ','
            << (event.bCommitStableByLag ? 1 : 0) << ','
            << (event.bMappedToObservedSpeech ? 1 : 0) << '\n';
    }
    return out.str();
}

static std::string commit_decisions_csv(const FOffgridAIAlignedVisemeTrack& track)
{
    std::ostringstream out;
    out << "event_index,pose,word,word_index,source_phone_index,source_phone_base,source_phone_class,commit_reason,alignment_reason,alignment_confidence,alignment_score_gap,alignment_observed_duration,alignment_expected_duration,aligned_phone_start,aligned_phone_end,render_start,render_center,render_end,commit_playback,commit_lead,commit_confidence_threshold,commit_alignable_lag,stable_by_confidence,stable_by_boundary,stable_by_duration,stable_by_lead,stable_by_lag,mapped_to_observed_speech\n";
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
            << to_std(event.AlignmentReason) << ','
            << event.AlignmentConfidence << ','
            << event.AlignmentScoreGap << ','
            << event.AlignmentObservedDurationSeconds << ','
            << event.AlignmentExpectedDurationSeconds << ','
            << event.AlignedPhoneStartSeconds << ','
            << event.AlignedPhoneEndSeconds << ','
            << event.RenderStartSeconds << ','
            << event.FinalRenderCenterSeconds << ','
            << event.RenderEndSeconds << ','
            << event.CommitPlaybackSeconds << ','
            << event.CommitLeadSeconds << ','
            << event.CommitConfidenceThreshold << ','
            << event.CommitAlignableLagSeconds << ','
            << (event.bCommitStableByConfidence ? 1 : 0) << ','
            << (event.bCommitStableByBoundary ? 1 : 0) << ','
            << (event.bCommitStableByDuration ? 1 : 0) << ','
            << (event.bCommitStableByLead ? 1 : 0) << ','
            << (event.bCommitStableByLag ? 1 : 0) << ','
            << (event.bMappedToObservedSpeech ? 1 : 0) << '\n';
    }
    return out.str();
}

static std::string alignment_csv(const FOffgridAITextVisemePlan& plan, const FOffgridAIOnlinePhoneAlignmentResult& alignment)
{
    std::ostringstream out;
    out << "phone_index,word_index,word_phone_index,word,phone_base,phone_class,start,center,end,match_score,score_gap,observed_duration,expected_duration,word_boundary_salience,visible_speech_seconds,visible_expected_seconds,speech_rate_scale,advance_reason\n";
    out << std::fixed << std::setprecision(6);
    for (int32 i = 0; i < plan.ExpectedPhones.Num(); ++i)
    {
        const auto& phone = plan.ExpectedPhones[i];
        out << i << ','
            << phone.WordIndex << ','
            << phone.WordPhoneIndex << ','
            << to_std(phone.SourceWord) << ','
            << to_std(phone.BasePhone) << ','
            << to_std(FOffgridAIOnlinePhoneAligner::PhoneClassToString(FOffgridAIOnlinePhoneAligner::ClassForPhoneBase(phone.BasePhone))) << ','
            << (alignment.PhoneStartSeconds.IsValidIndex(i) ? alignment.PhoneStartSeconds[i] : -1.0f) << ','
            << (alignment.PhoneCenterSeconds.IsValidIndex(i) ? alignment.PhoneCenterSeconds[i] : -1.0f) << ','
            << (alignment.PhoneEndSeconds.IsValidIndex(i) ? alignment.PhoneEndSeconds[i] : -1.0f) << ','
            << (alignment.PhoneMatchScores.IsValidIndex(i) ? alignment.PhoneMatchScores[i] : 0.0f) << ','
            << (alignment.PhoneScoreGaps.IsValidIndex(i) ? alignment.PhoneScoreGaps[i] : 0.0f) << ','
            << (alignment.PhoneObservedDurations.IsValidIndex(i) ? alignment.PhoneObservedDurations[i] : 0.0f) << ','
            << (alignment.PhoneExpectedDurations.IsValidIndex(i) ? alignment.PhoneExpectedDurations[i] : 0.0f) << ','
            << (alignment.PhoneWordBoundarySalience.IsValidIndex(i) ? alignment.PhoneWordBoundarySalience[i] : 0.0f) << ','
            << alignment.VisibleSpeechSeconds << ','
            << alignment.VisibleExpectedSeconds << ','
            << alignment.SpeechRateScale << ','
            << to_std(alignment.PhoneAdvanceReasons.IsValidIndex(i) ? alignment.PhoneAdvanceReasons[i] : NAME_None) << '\n';
    }
    return out.str();
}

static std::string runtime_phone_alignment_csv(const FOffgridAITextVisemePlan& plan, const FOffgridAIAlignedVisemeTrack& track)
{
    std::ostringstream out;
    out << "phone_index,word_index,word_phone_index,word,phone_base,phone_class,start,center,end,mapped_to_observed_speech\n";
    out << std::fixed << std::setprecision(6);
    for (int32 i = 0; i < plan.ExpectedPhones.Num(); ++i)
    {
        const auto& phone = plan.ExpectedPhones[i];
        const float start = track.RuntimeObservedPhoneStartSeconds.IsValidIndex(i)
            ? track.RuntimeObservedPhoneStartSeconds[i]
            : -1.0f;
        const float end = track.RuntimeObservedPhoneEndSeconds.IsValidIndex(i)
            ? track.RuntimeObservedPhoneEndSeconds[i]
            : -1.0f;
        const bool bObserved = start >= 0.0f && end > start;
        const float center = (start >= 0.0f && end > start) ? (start + end) * 0.5f : -1.0f;
        out << i << ','
            << phone.WordIndex << ','
            << phone.WordPhoneIndex << ','
            << to_std(phone.SourceWord) << ','
            << to_std(phone.BasePhone) << ','
            << to_std(FOffgridAIOnlinePhoneAligner::PhoneClassToString(FOffgridAIOnlinePhoneAligner::ClassForPhoneBase(phone.BasePhone))) << ','
            << start << ','
            << center << ','
            << end << ','
            << (bObserved ? 1 : 0) << '\n';
    }
    return out.str();
}

static FName ResolvePlannedPoseForHarness(const FOffgridAITextVisemeEvent& event)
{
    return event.PoseID.IsNone() ? FName(FOffgridAITextVisemePlanner::ToPoseKey(event.Viseme)) : event.PoseID;
}

static FOffgridAIAlignedVisemeTrack build_direct_aligner_track(
    const FOffgridAITextVisemePlan& plan,
    const FOffgridAIOnlinePhoneAlignmentResult& alignment)
{
    FOffgridAIAlignedVisemeTrack out;
    out.LineID = FName(TEXT("direct_aligner_batch"));
    out.NPCID = FName(TEXT("liplab"));

    for (int32 event_index = 0; event_index < plan.Events.Num(); ++event_index)
    {
        const FOffgridAITextVisemeEvent& src = plan.Events[event_index];
        int32 phone_index = src.SourcePhoneGlobalIndex;
        if (phone_index == INDEX_NONE)
        {
            phone_index = FOffgridAIOnlinePhoneAligner::FindPhoneForEvent(plan, src);
        }
        if (!alignment.PhoneStartSeconds.IsValidIndex(phone_index)
            || !alignment.PhoneEndSeconds.IsValidIndex(phone_index)
            || alignment.PhoneStartSeconds[phone_index] < 0.0f
            || alignment.PhoneEndSeconds[phone_index] <= alignment.PhoneStartSeconds[phone_index])
        {
            continue;
        }

        const float start = alignment.PhoneStartSeconds[phone_index];
        const float end = alignment.PhoneEndSeconds[phone_index];
        const float center = alignment.PhoneCenterSeconds.IsValidIndex(phone_index) && alignment.PhoneCenterSeconds[phone_index] >= 0.0f
            ? alignment.PhoneCenterSeconds[phone_index]
            : (start + end) * 0.5f;

        FOffgridAIAlignedVisemeEvent e;
        e.EventIndex = event_index;
        e.PoseID = ResolvePlannedPoseForHarness(src);
        e.Strength = src.Strength;
        e.SourceWord = src.SourceText;
        e.WordIndex = src.WordIndex;
        e.PhraseIndex = src.PhraseIndex;
        e.SentenceIndex = src.SentenceIslandIndex;
        e.bIsStrongVisibleEvent = src.bIsStrongVisibleEvent;
        e.TextCenterNorm = (src.StartNorm + src.EndNorm) * 0.5f;
        e.FinalRenderCenterSeconds = center;
        e.RenderStartSeconds = start;
        e.RenderEndSeconds = end;
        e.SourcePhoneIndex = phone_index;
        e.SourcePhoneBase = src.SourcePhoneBase;
        e.SourcePhoneClass = FName(*FOffgridAIOnlinePhoneAligner::PhoneClassToString(FOffgridAIOnlinePhoneAligner::ClassForPhoneBase(src.SourcePhoneBase)));
        e.AlignedPhoneStartSeconds = start;
        e.AlignedPhoneEndSeconds = end;
        e.AlignmentConfidence = alignment.PhoneMatchScores.IsValidIndex(phone_index) ? alignment.PhoneMatchScores[phone_index] : 0.0f;
        e.AlignmentScoreGap = alignment.PhoneScoreGaps.IsValidIndex(phone_index) ? alignment.PhoneScoreGaps[phone_index] : 0.0f;
        e.AlignmentObservedDurationSeconds = alignment.PhoneObservedDurations.IsValidIndex(phone_index) ? alignment.PhoneObservedDurations[phone_index] : (end - start);
        e.AlignmentExpectedDurationSeconds = alignment.PhoneExpectedDurations.IsValidIndex(phone_index) ? alignment.PhoneExpectedDurations[phone_index] : 0.0f;
        e.AlignmentReason = FName(TEXT("direct_batch_aligner"));
        e.bMappedToObservedSpeech = true;
        e.CommitReason = FName(TEXT("direct_batch_aligner"));
        out.Events.Add(e);
    }

    if (out.Events.Num() > 0)
    {
        out.SpeechStartSeconds = out.Events[0].RenderStartSeconds;
        out.SpeechEndSeconds = out.Events.Last().RenderEndSeconds;
    }
    return out;
}

static std::string occupancy_diagnostics_csv(const TArray<FOffgridAIAudioOccupancyDiagnosticRow>& rows)
{
    std::ostringstream out;
    out << "line_id,update_ordinal,final_replay,current_playback,preroll,event_index,word,pose,planned_center,committed_center,render_start,render_end,commit_reason,playback_mode,aligned_phone_start,aligned_phone_end,audio_active,text_playhead,required_active_elapsed,observed_active_elapsed,active_progress_deficit,required_progress_norm,observed_progress_norm,active_progress_ratio,mapped_to_observed_speech,diagnostic_kind\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& row : rows)
    {
        out << to_std(row.LineID) << ','
            << row.UpdateOrdinal << ','
            << (row.bFinalReplay ? 1 : 0) << ','
            << row.CurrentPlaybackSec << ','
            << row.PrerollSec << ','
            << row.SourceEventIndex << ','
            << to_std(row.Word) << ','
            << to_std(row.PoseID) << ','
            << row.PlannedCenterSec << ','
            << row.CommittedCenterSec << ','
            << row.RenderStartSec << ','
            << row.RenderEndSec << ','
            << to_std(row.CommitReason) << ','
            << to_std(row.PlaybackMode) << ','
            << row.AlignedPhoneStartSec << ','
            << row.AlignedPhoneEndSec << ','
            << row.AudioActiveSec << ','
            << row.TextPlayheadSec << ','
            << row.RequiredActiveElapsedSec << ','
            << row.ObservedActiveElapsedSec << ','
            << row.ActiveProgressDeficitSec << ','
            << row.RequiredProgressNorm << ','
            << row.ObservedProgressNorm << ','
            << row.ActiveProgressRatio << ','
            << (row.bMappedToObservedSpeech ? 1 : 0) << ','
            << to_std(row.DiagnosticKind) << '\n';
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

static std::vector<TimeSpan> build_predicted_speech_regions(const FOffgridAIAlignedVisemeTrack& track)
{
    std::vector<TimeSpan> spans;
    constexpr double merge_gap_seconds = 0.120;
    for (const auto& event : track.Events)
    {
        const double start = static_cast<double>(event.RenderStartSeconds);
        const double end = static_cast<double>(event.RenderEndSeconds);
        if (spans.empty() || start > spans.back().end + merge_gap_seconds)
        {
            spans.push_back(TimeSpan{static_cast<int>(spans.size()), start, end});
        }
        else
        {
            spans.back().end = std::max(spans.back().end, end);
        }
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
        const double error_ms = std::abs(static_cast<double>(event_it->RenderStartSeconds) - gold_it->start) * 1000.0;
        errors.push_back(error_ms);
        early_sum += std::max(0.0, gold_it->start - static_cast<double>(event_it->RenderStartSeconds)) * 1000.0;
        late_sum += std::max(0.0, static_cast<double>(event_it->RenderStartSeconds) - gold_it->start) * 1000.0;
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
        build_predicted_speech_regions(track));
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
    out << "word_index,word,gold_word_start,gold_word_end,first_gold_pose,first_gold_start,first_event_pose,first_event_start,error_ms,missing_event\n";
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
            out << ",,0,true\n";
        }
        else
        {
            const double event_start = static_cast<double>(event_it->RenderStartSeconds);
            out << to_std(event_it->PoseID) << "," << event_start << ","
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

static float wav_duration_seconds(const Wav& wav)
{
    if (wav.sample_rate <= 0 || wav.channels <= 0) return 0.0f;
    return static_cast<float>(wav.samples.size()) / static_cast<float>(wav.sample_rate * wav.channels);
}

static StreamConfig parse_stream_config(int argc, char** argv)
{
    StreamConfig config;
    for (int i = 2; i < argc; ++i)
    {
        const std::string arg = argv[i];
        auto parse_ms = [&](const std::string& value, const char* flag) {
            const double parsed = std::stod(value);
            if (parsed < 0.0)
            {
                throw std::runtime_error(std::string(flag) + " must be non-negative");
            }
            return static_cast<float>(parsed / 1000.0);
        };

        if (arg == "--buffer-ms")
        {
            if (i + 1 >= argc) throw std::runtime_error("--buffer-ms requires a value");
            config.buffer_seconds = parse_ms(argv[++i], "--buffer-ms");
        }
        else if (arg == "--chunk-ms")
        {
            if (i + 1 >= argc) throw std::runtime_error("--chunk-ms requires a value");
            config.chunk_seconds = parse_ms(argv[++i], "--chunk-ms");
        }
        else
        {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (config.chunk_seconds <= 0.0f)
    {
        throw std::runtime_error("--chunk-ms must be greater than zero");
    }
    return config;
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
    const int32 frames_per_chunk = stream_chunk_frames(wav.sample_rate, stream.chunk_seconds);
    const int32 total_frames = static_cast<int32>(wav.samples.size() / static_cast<size_t>(wav.channels));
    int32 frame_cursor = 0;

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
        const float playback_seconds = std::max(0.0f, observed_end - buffer_seconds);
        session.Update(playback_seconds);

        frame_cursor += chunk_frames;
    }

    session.CloseInputStream();

    const float duration_seconds = wav_duration_seconds(wav);
    float playback_seconds = std::max(0.0f, duration_seconds - buffer_seconds);
    const float tick_seconds = 0.020f;
    while (playback_seconds + tick_seconds < duration_seconds)
    {
        playback_seconds += tick_seconds;
        session.Update(playback_seconds);
    }

    session.Finalize(duration_seconds);
}

int main(int argc, char** argv)
{
    try
    {
        using clock = std::chrono::steady_clock;
        fs::path root = argc > 1 ? argv[1] : fs::current_path();
        const StreamConfig stream = parse_stream_config(argc, argv);
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
            const auto& committed = session.GetCommittedTrack();

            const fs::path case_dir = out_root / stem;
            fs::create_directories(case_dir);
            write_text(case_dir / "planned.csv", planned_csv(plan));
            write_text(case_dir / "speech_regions.csv", speech_csv(speech));
            write_text(case_dir / "committed.csv", committed_csv(committed));
            write_text(case_dir / "commit_decisions.csv", commit_decisions_csv(committed));
            write_text(case_dir / "occupancy_diagnostics.csv", occupancy_diagnostics_csv(session.GetAudioOccupancyDiagnosticRows()));
            write_text(case_dir / "stream_tail.csv", stream_tail_csv(session.GetStreamTailDiagnosticRow()));
            write_text(case_dir / "runtime_phone_alignment.csv", runtime_phone_alignment_csv(plan, committed));

            FOffgridAIOnlinePhoneAlignmentInput alignment_input;
            alignment_input.Plan = &plan;
            alignment_input.AudioFeatureFrames = &session.GetAudioFeatureFrames();
            alignment_input.SpeechIslands = &speech;
            alignment_input.ObservedAudioEndSec = wav_duration_seconds(wav);
            alignment_input.PlaybackSec = wav_duration_seconds(wav);
            alignment_input.LookaheadSec = stream.buffer_seconds;
            alignment_input.CommitLagSec = 0.120f;
            alignment_input.bFinal = true;
            const auto final_alignment = FOffgridAIOnlinePhoneAligner::Compute(alignment_input);
            write_text(case_dir / "online_phone_alignment.csv", alignment_csv(plan, final_alignment));
            const auto direct_aligner_track = build_direct_aligner_track(plan, final_alignment);
            write_text(case_dir / "direct_aligner_visemes.csv", committed_csv(direct_aligner_track));

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
                const auto gold_word_boundaries = gold_internal_word_boundary_times(gold_words);
                const auto audio_boundary_candidates = detect_audio_word_boundary_candidates(session.GetAudioFeatureFrames(), speech, gold_word_boundaries);
                const auto audio_boundary_eval = evaluate_audio_word_boundary_candidates(audio_boundary_candidates, gold_word_boundaries);
                write_text(case_dir / "audio_word_boundary_candidates.csv", audio_word_boundary_candidates_csv(audio_boundary_candidates));
                write_text(case_dir / "audio_word_boundary_grade.json", audio_word_boundary_grade_json(audio_boundary_eval));
                gold_viseme_count = handmade.size();
                report = grade(committed, handmade, gold_words, gold_speech);
                const GradeReport direct_aligner_report = grade(direct_aligner_track, handmade, gold_words, gold_speech);
                write_text(case_dir / "direct_aligner_grade.json", grade_json(direct_aligner_report));
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
