#include "../src/lipsync/LipsyncCore.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <cstring>

using namespace offgrid::lipsync;
namespace fs = std::filesystem;

struct Wav { int sample_rate = 0; int channels = 0; std::vector<int16_t> samples; };

static std::string read_text(const fs::path& p) { std::ifstream f(p); std::ostringstream ss; ss << f.rdbuf(); return ss.str(); }
static void write_text(const fs::path& p, const std::string& s) { fs::create_directories(p.parent_path()); std::ofstream f(p); f << s; }

template<class T> static T read_le(const std::vector<uint8_t>& b, size_t off) { T v{}; std::memcpy(&v, b.data()+off, sizeof(T)); return v; }

static Wav read_wav(const fs::path& p) {
    std::ifstream f(p, std::ios::binary); if (!f) throw std::runtime_error("cannot open wav: " + p.string());
    std::vector<uint8_t> b((std::istreambuf_iterator<char>(f)), {});
    if (b.size() < 44 || std::memcmp(b.data(), "RIFF", 4) || std::memcmp(b.data()+8, "WAVE", 4)) throw std::runtime_error("bad wav: " + p.string());
    Wav w; size_t pos = 12; size_t data_off = 0, data_size = 0; uint16_t bits = 0, fmt = 0;
    while (pos + 8 <= b.size()) {
        char id[5]{}; std::memcpy(id, b.data()+pos, 4); uint32_t sz = read_le<uint32_t>(b, pos+4); pos += 8;
        if (std::string(id) == "fmt ") { fmt = read_le<uint16_t>(b,pos); w.channels = read_le<uint16_t>(b,pos+2); w.sample_rate = read_le<uint32_t>(b,pos+4); bits = read_le<uint16_t>(b,pos+14); }
        else if (std::string(id) == "data") { data_off = pos; data_size = sz; }
        pos += sz + (sz & 1);
    }
    if (fmt != 1 || bits != 16 || !data_off) throw std::runtime_error("only PCM16 wav supported: " + p.string());
    size_t n = data_size / 2; w.samples.resize(n); std::memcpy(w.samples.data(), b.data()+data_off, n*2); return w;
}

static std::vector<HandmadeLabel> read_handmade_csv(const fs::path& p) {
    std::vector<HandmadeLabel> out; std::ifstream f(p); if (!f) return out;
    std::string line; std::getline(f,line);
    while (std::getline(f,line)) {
        if (line.empty()) continue; std::stringstream ss(line); std::string cell; HandmadeLabel h;
        std::getline(ss, cell, ','); h.start = std::stod(cell);
        std::getline(ss, cell, ','); h.end = std::stod(cell);
        std::getline(ss, h.pose, ',');
        std::getline(ss, h.word, ',');
        if (std::getline(ss, cell, ',')) h.confidence = std::stod(cell);
        out.push_back(h);
    }
    return out;
}

static std::string planned_csv(const VisemePlan& p) {
    std::ostringstream o; o << "index,pose,word,word_index,phrase_index,sentence_index,text_center_norm,strength\n";
    for (auto& e : p.events) o << e.index << ',' << e.pose << ',' << e.word << ',' << e.word_index << ',' << e.phrase_index << ',' << e.sentence_index << ',' << e.text_center_norm << ',' << e.strength << '\n';
    return o.str();
}
static std::string speech_csv(const std::vector<SpeechRegion>& rs) {
    std::ostringstream o; o << "index,start,end,energy\n" << std::fixed << std::setprecision(6);
    for (auto& r : rs) o << r.index << ',' << r.start << ',' << r.end << ',' << r.energy << '\n'; return o.str();
}
static std::string committed_csv(const std::vector<CommittedVisemeEvent>& es) {
    std::ostringstream o; o << "index,start,center,end,pose,word,word_index,phrase_index,sentence_index,strength,reason\n" << std::fixed << std::setprecision(6);
    for (auto& e : es) o << e.index << ',' << e.start << ',' << e.center << ',' << e.end << ',' << e.pose << ',' << e.word << ',' << e.word_index << ',' << e.phrase_index << ',' << e.sentence_index << ',' << e.strength << ',' << e.reason << '\n'; return o.str();
}
static std::string grade_json(const GradeReport& g) {
    std::ostringstream o; o << std::fixed << std::setprecision(3);
    o << "{\n  \"reference_count\": " << g.reference_count << ",\n  \"committed_count\": " << g.committed_count << ",\n  \"matched_count\": " << g.matched_count << ",\n  \"missing_count\": " << g.missing_count << ",\n  \"extra_count\": " << g.extra_count << ",\n  \"order_violations\": " << g.order_violations << ",\n  \"mean_abs_center_error_ms\": " << g.mean_abs_center_error_ms << ",\n  \"p90_abs_center_error_ms\": " << g.p90_abs_center_error_ms << ",\n  \"early_start_rate\": " << g.early_start_rate << ",\n  \"late_tail_rate\": " << g.late_tail_rate << "\n}\n"; return o.str();
}

int main(int argc, char** argv) {
    try {
        fs::path root = argc > 1 ? argv[1] : fs::current_path();
        fs::path out_root = root / "outputs" / "runs" / "latest"; fs::remove_all(out_root); fs::create_directories(out_root);
        int cases = 0; bool all_ok = true;
        for (auto& ent : fs::directory_iterator(root / "inputs" / "transcripts")) {
            if (!ent.is_regular_file() || ent.path().extension() != ".txt") continue;
            std::string stem = ent.path().stem().string(); fs::path wav_path = root / "inputs" / "wav" / (stem + ".wav");
            if (!fs::exists(wav_path)) { std::cerr << "missing wav for " << stem << "\n"; all_ok = false; continue; }
            auto transcript = read_text(ent.path()); auto wav = read_wav(wav_path);
            auto plan = build_plan(transcript); auto speech = detect_speech_regions(wav.samples, wav.sample_rate, wav.channels); auto committed = schedule_visemes(plan, speech);
            fs::path case_dir = out_root / stem; fs::create_directories(case_dir);
            write_text(case_dir / "planned.csv", planned_csv(plan)); write_text(case_dir / "speech_regions.csv", speech_csv(speech)); write_text(case_dir / "committed.csv", committed_csv(committed));
            auto handmade = read_handmade_csv(root / "inputs" / "handmade" / (stem + ".csv"));
            GradeReport gr = grade(committed, handmade); write_text(case_dir / "grade.json", grade_json(gr));
            std::cout << stem << ": planned=" << plan.events.size() << " committed=" << committed.size() << " handmade=" << handmade.size() << " mean_ms=" << gr.mean_abs_center_error_ms << "\n";
            ++cases;
        }
        if (cases == 0) { std::cerr << "no transcript cases found\n"; return 2; }
        return all_ok ? 0 : 1;
    } catch (const std::exception& e) { std::cerr << "error: " << e.what() << "\n"; return 1; }
}
