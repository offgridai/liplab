#include "LipsyncCore.h"
#include <cmath>
#include <cctype>
#include <numeric>
#include <sstream>

namespace offgrid::lipsync {
namespace {
std::string lower_word(const std::string& s) {
    std::string out;
    for (unsigned char c : s) {
        if (std::isalnum(c) || c == '\'') out.push_back(static_cast<char>(std::tolower(c)));
    }
    return out;
}

bool is_boundary(char c) { return c == '.' || c == '!' || c == '?' || c == ',' || c == ';' || c == ':'; }

std::vector<std::string> phones_for_word(const std::string& w) {
    static const std::map<std::string, std::vector<std::string>> dict = {
        {"welcome", {"W","EH","L","K","AH","M"}}, {"to", {"T","UW"}}, {"alfie's", {"AE","L","F","IY","Z"}},
        {"bodega", {"B","OW","D","EY","G","AH"}}, {"how", {"HH","AW"}}, {"can", {"K","AE","N"}},
        {"i", {"AY"}}, {"help", {"HH","EH","L","P"}}, {"you", {"Y","UW"}}, {"today", {"T","AH","D","EY"}},
        {"we've", {"W","IY","V"}}, {"got", {"G","AA","T"}}, {"turkey", {"T","ER","K","IY"}},
        {"ham", {"HH","AE","M"}}, {"and", {"AE","N","D"}}, {"veggie", {"V","EH","JH","IY"}},
        {"which", {"W","IH","CH"}}, {"one", {"W","AH","N"}}, {"sounds", {"S","AW","N","D","Z"}}, {"good", {"G","UH","D"}}
    };
    auto it = dict.find(w);
    if (it != dict.end()) return it->second;
    std::vector<std::string> phones;
    for (char c : w) {
        switch (c) {
            case 'm': case 'b': case 'p': phones.push_back("M"); break;
            case 'f': case 'v': case 's': case 'z': phones.push_back("F"); break;
            case 'w': phones.push_back("W"); break;
            case 'o': case 'u': phones.push_back("OW"); break;
            case 'e': case 'i': case 'y': phones.push_back("IY"); break;
            case 'a': phones.push_back("AE"); break;
            default: break;
        }
    }
    if (phones.empty()) phones.push_back("AH");
    return phones;
}

std::string pose_for_phone(const std::string& p) {
    if (p == "M" || p == "B" || p == "P") return "22_MBP";
    if (p == "F" || p == "V" || p == "S" || p == "Z") return "20_FV";
    if (p == "W" || p == "Y") return "12_Ww-Oo-";
    if (p == "UW" || p == "UH" || p == "OW" || p == "OY") return "11_Oo";
    if (p == "IY" || p == "IH" || p == "EY") return "03_Ee";
    if (p == "EH" || p == "AE" || p == "ER") return "06_Eh";
    if (p == "AA" || p == "AH" || p == "AO" || p == "AW" || p == "AY") return "07_Aa";
    if (p == "CH" || p == "JH" || p == "SH" || p == "ZH") return "14_ChJjSh";
    return "";
}

bool strong_pose(const std::string& p) { return p == "22_MBP" || p == "20_FV" || p == "14_ChJjSh" || p == "12_Ww-Oo-"; }
bool closure_pose(const std::string& p) { return p == "22_MBP"; }
bool fricative_pose(const std::string& p) { return p == "20_FV" || p == "14_ChJjSh"; }

double span_for_pose(const std::string& p, const RuntimeSettings& s) {
    if (p == "22_MBP") return 0.105;
    if (p == "20_FV" || p == "14_ChJjSh") return 0.095;
    return std::clamp(0.090, s.min_event_span, s.max_event_span);
}

} // namespace

VisemePlan build_plan(const std::string& transcript) {
    VisemePlan plan; plan.transcript = transcript;
    std::vector<std::pair<std::string, char>> words;
    std::string cur;
    char pending = 0;
    for (char c : transcript) {
        if (std::isspace(static_cast<unsigned char>(c)) || is_boundary(c)) {
            if (!cur.empty()) { words.push_back({lower_word(cur), pending}); cur.clear(); pending = 0; }
            if (is_boundary(c) && !words.empty()) words.back().second = c;
        } else cur.push_back(c);
    }
    if (!cur.empty()) words.push_back({lower_word(cur), pending});

    int phrase = 0, sentence = 0, index = 0;
    std::vector<std::tuple<std::string,int,int,int,std::string>> visible;
    for (size_t wi = 0; wi < words.size(); ++wi) {
        auto phones = phones_for_word(words[wi].first);
        std::string last_pose;
        for (const auto& ph : phones) {
            std::string pose = pose_for_phone(ph);
            if (!pose.empty() && pose != last_pose) {
                visible.push_back({pose, static_cast<int>(wi), phrase, sentence, words[wi].first});
                last_pose = pose;
            }
        }
        if (words[wi].second == ',' || words[wi].second == ';' || words[wi].second == ':') ++phrase;
        if (words[wi].second == '.' || words[wi].second == '!' || words[wi].second == '?') { ++sentence; ++phrase; }
    }
    const float denom = static_cast<float>(std::max<size_t>(visible.size(), 1));
    for (size_t i = 0; i < visible.size(); ++i) {
        VisemePlanEvent e;
        e.index = index++;
        e.pose = std::get<0>(visible[i]);
        e.word_index = std::get<1>(visible[i]);
        e.phrase_index = std::get<2>(visible[i]);
        e.sentence_index = std::get<3>(visible[i]);
        e.word = std::get<4>(visible[i]);
        e.text_center_norm = (static_cast<float>(i) + 0.5f) / denom;
        e.strength = strong_pose(e.pose) ? 0.9f : 0.68f;
        plan.events.push_back(e);
    }
    return plan;
}

std::vector<SpeechRegion> detect_speech_regions(const std::vector<int16_t>& pcm, int sample_rate, int channels) {
    std::vector<SpeechRegion> regions;
    if (sample_rate <= 0 || channels <= 0 || pcm.empty()) return regions;
    const int frame_samples = std::max(1, sample_rate / 100); // 10 ms mono frames
    std::vector<double> rms;
    for (size_t pos = 0; pos + channels <= pcm.size(); pos += static_cast<size_t>(frame_samples * channels)) {
        double sum = 0; int n = 0;
        for (int i = 0; i < frame_samples && pos + static_cast<size_t>(i * channels) < pcm.size(); ++i) {
            double mono = 0;
            for (int ch = 0; ch < channels; ++ch) mono += pcm[pos + static_cast<size_t>(i * channels + ch)] / 32768.0;
            mono /= channels; sum += mono * mono; ++n;
        }
        rms.push_back(n ? std::sqrt(sum / n) : 0.0);
    }
    double mean = rms.empty() ? 0.0 : std::accumulate(rms.begin(), rms.end(), 0.0) / rms.size();
    double threshold = std::max(0.012, mean * 0.42);
    bool active = false; int start = 0; double energy = 0;
    auto finish = [&](int end_frame) {
        double st = start * 0.01, en = end_frame * 0.01;
        if (en - st >= 0.045) regions.push_back({static_cast<int>(regions.size()), st, en, energy / std::max(1, end_frame - start)});
    };
    for (int i = 0; i < static_cast<int>(rms.size()); ++i) {
        bool speech = rms[i] >= threshold;
        if (speech && !active) { active = true; start = i; energy = 0; }
        if (active) energy += rms[i];
        if (!speech && active) { finish(i); active = false; }
    }
    if (active) finish(static_cast<int>(rms.size()));
    // merge short internal gaps, useful for syllabic TTS with tiny valleys
    std::vector<SpeechRegion> merged;
    for (auto r : regions) {
        if (!merged.empty() && r.start - merged.back().end <= 0.080) {
            merged.back().end = r.end;
            merged.back().energy = std::max(merged.back().energy, r.energy);
        } else { r.index = static_cast<int>(merged.size()); merged.push_back(r); }
    }
    return merged;
}

std::vector<CommittedVisemeEvent> schedule_visemes(const VisemePlan& plan, const std::vector<SpeechRegion>& input_regions, const RuntimeSettings& s) {
    std::vector<CommittedVisemeEvent> out;
    if (plan.events.empty()) return out;
    std::vector<SpeechRegion> regions = input_regions;
    if (regions.empty()) regions.push_back({0, 0.0, s.fallback_duration_seconds, 0.0});
    // group by sentence first to avoid whole-line compression after pauses/lists
    int min_sentence = plan.events.front().sentence_index;
    int max_sentence = plan.events.back().sentence_index;
    size_t region_cursor = 0;
    for (int sent = min_sentence; sent <= max_sentence; ++sent) {
        std::vector<const VisemePlanEvent*> evs;
        for (const auto& e : plan.events) if (e.sentence_index == sent) evs.push_back(&e);
        if (evs.empty()) continue;
        size_t remaining_sentences = static_cast<size_t>(max_sentence - sent + 1);
        size_t regions_left = regions.size() - std::min(region_cursor, regions.size());
        size_t take = std::max<size_t>(1, regions_left / remaining_sentences);
        if (sent == max_sentence) take = regions_left;
        if (take == 0) take = 1;
        double start = regions[std::min(region_cursor, regions.size()-1)].start;
        double end = regions[std::min(region_cursor + take - 1, regions.size()-1)].end;
        region_cursor = std::min(region_cursor + take, regions.size());
        double dur = std::max(0.001, end - start);
        // phrase-local distribution inside the sentence prevents list items from crushing the tail.
        int min_phrase = evs.front()->phrase_index;
        int max_phrase = evs.back()->phrase_index;
        for (int ph = min_phrase; ph <= max_phrase; ++ph) {
            std::vector<const VisemePlanEvent*> pevs;
            for (auto* e : evs) if (e->phrase_index == ph) pevs.push_back(e);
            if (pevs.empty()) continue;
            double p0 = start + dur * ((ph - min_phrase) / static_cast<double>(std::max(1, max_phrase - min_phrase + 1)));
            double p1 = start + dur * ((ph - min_phrase + 1) / static_cast<double>(std::max(1, max_phrase - min_phrase + 1)));
            const double pdur = std::max(0.001, p1 - p0);
            for (size_t i = 0; i < pevs.size(); ++i) {
                const auto& e = *pevs[i];
                double center = p0 + pdur * ((static_cast<double>(i) + 0.5) / pevs.size());
                if (closure_pose(e.pose)) center -= s.closure_lead;
                else if (fricative_pose(e.pose)) center -= s.fricative_lead;
                double span = span_for_pose(e.pose, s);
                CommittedVisemeEvent c;
                c.index = e.index; c.pose = e.pose; c.word = e.word; c.word_index = e.word_index;
                c.phrase_index = e.phrase_index; c.sentence_index = e.sentence_index; c.strength = e.strength;
                c.center = std::clamp(center, p0 + 0.010, p1 - 0.010);
                c.start = std::max(p0, c.center - span * 0.5);
                c.end = std::min(p1, c.center + span * 0.5);
                c.reason = "sentence_phrase_active_coverage";
                if (!out.empty() && c.center <= out.back().center) {
                    c.center = out.back().center + 0.012;
                    c.start = std::max(c.start, out.back().center + 0.002);
                    c.end = std::max(c.end, c.center + 0.020);
                    c.reason += "+monotonic_guard";
                }
                out.push_back(c);
            }
        }
    }
    return out;
}

std::map<std::string, float> sample_pose(const std::vector<CommittedVisemeEvent>& events, double t) {
    std::map<std::string, float> weights;
    for (const auto& e : events) {
        if (t < e.start || t > e.end || e.end <= e.start) continue;
        double half = std::max(0.001, (e.end - e.start) * 0.5);
        double x = std::abs(t - e.center) / half;
        float w = static_cast<float>(std::max(0.0, 1.0 - x) * e.strength);
        weights[e.pose] = std::max(weights[e.pose], w);
    }
    return weights;
}

GradeReport grade(const std::vector<CommittedVisemeEvent>& committed, const std::vector<HandmadeLabel>& handmade) {
    GradeReport r; r.reference_count = static_cast<int>(handmade.size()); r.committed_count = static_cast<int>(committed.size());
    std::vector<double> errors;
    std::vector<bool> used(committed.size(), false);
    int last_match = -1;
    for (const auto& h : handmade) {
        double hc = (h.start + h.end) * 0.5;
        int best = -1; double best_err = 1e9;
        for (size_t i = 0; i < committed.size(); ++i) {
            if (used[i] || committed[i].pose != h.pose) continue;
            double err = std::abs(committed[i].center - hc);
            if (err < best_err) { best_err = err; best = static_cast<int>(i); }
        }
        if (best >= 0 && best_err <= 0.180) {
            used[best] = true; ++r.matched_count; errors.push_back(best_err * 1000.0);
            if (best < last_match) ++r.order_violations;
            last_match = best;
            if (committed[best].start + 0.050 < h.start) r.early_start_rate += 1.0;
            if (committed[best].end - 0.050 > h.end) r.late_tail_rate += 1.0;
        }
    }
    r.missing_count = r.reference_count - r.matched_count;
    r.extra_count = r.committed_count - r.matched_count;
    if (!errors.empty()) {
        r.mean_abs_center_error_ms = std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
        std::sort(errors.begin(), errors.end());
        r.p90_abs_center_error_ms = errors[std::min(errors.size()-1, static_cast<size_t>(std::floor(errors.size()*0.9)))];
        r.early_start_rate /= errors.size(); r.late_tail_rate /= errors.size();
    }
    return r;
}

} // namespace offgrid::lipsync
