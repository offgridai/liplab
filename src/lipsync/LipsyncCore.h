#pragma once
#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace offgrid::lipsync {

struct VisemePlanEvent {
    int index = -1;
    std::string pose;
    std::string word;
    int word_index = -1;
    int phrase_index = 0;
    int sentence_index = 0;
    float text_center_norm = 0.0f;
    float strength = 1.0f;
};

struct VisemePlan {
    std::string transcript;
    std::vector<VisemePlanEvent> events;
};

struct SpeechRegion {
    int index = -1;
    double start = 0.0;
    double end = 0.0;
    double energy = 0.0;
};

struct CommittedVisemeEvent {
    int index = -1;
    std::string pose;
    std::string word;
    int word_index = -1;
    int phrase_index = 0;
    int sentence_index = 0;
    double start = 0.0;
    double center = 0.0;
    double end = 0.0;
    float strength = 1.0f;
    std::string reason;
};

struct HandmadeLabel {
    double start = 0.0;
    double end = 0.0;
    std::string pose;
    std::string word;
    double confidence = 1.0;
};

struct GradeReport {
    int reference_count = 0;
    int committed_count = 0;
    int matched_count = 0;
    int missing_count = 0;
    int extra_count = 0;
    int order_violations = 0;
    double mean_abs_center_error_ms = 0.0;
    double p90_abs_center_error_ms = 0.0;
    double early_start_rate = 0.0;
    double late_tail_rate = 0.0;
};

struct RuntimeSettings {
    double min_event_span = 0.075;
    double max_event_span = 0.140;
    double closure_lead = 0.030;
    double fricative_lead = 0.020;
    double soft_gap_merge_seconds = 0.130;
    double fallback_duration_seconds = 0.650;
};

VisemePlan build_plan(const std::string& transcript);
std::vector<SpeechRegion> detect_speech_regions(const std::vector<int16_t>& pcm, int sample_rate, int channels);
std::vector<CommittedVisemeEvent> schedule_visemes(const VisemePlan& plan, const std::vector<SpeechRegion>& regions, const RuntimeSettings& settings = {});
std::map<std::string, float> sample_pose(const std::vector<CommittedVisemeEvent>& events, double time_seconds);
GradeReport grade(const std::vector<CommittedVisemeEvent>& committed, const std::vector<HandmadeLabel>& handmade);

} // namespace offgrid::lipsync
