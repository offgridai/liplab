#include "Lipsync/OffgridAILipsyncRuntimeAdapter.h"
#include "Lipsync/OffgridAIVisemePerformer.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
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
};

struct GradeReport
{
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

struct StreamConfig
{
    float buffer_seconds = 0.350f;
    float chunk_seconds = 0.040f;
};

static std::string to_std(const FString& value) { return value.Str(); }
static std::string to_std(const FName& value) { return value.Str(); }

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
        std::stringstream ss(line);
        std::string cell;
        HandmadeLabel label;
        std::getline(ss, cell, ',');
        label.start = std::stod(cell);
        std::getline(ss, cell, ',');
        label.end = std::stod(cell);
        std::getline(ss, label.pose, ',');
        std::getline(ss, label.word, ',');
        if (std::getline(ss, cell, ',')) label.confidence = std::stod(cell);
        out.push_back(label);
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
    out << "index,start,center,end,pose,word,word_index,phrase_index,sentence_index,strength,reason,alignment_reason,source_phone_class,mapped_to_observed_speech\n";
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
            << to_std(event.SourcePhoneClass) << ','
            << (event.bMappedToObservedSpeech ? 1 : 0) << '\n';
    }
    return out.str();
}

static GradeReport grade(const FOffgridAIAlignedVisemeTrack& track, const std::vector<HandmadeLabel>& handmade)
{
    GradeReport report;
    report.reference_count = static_cast<int>(handmade.size());
    report.committed_count = track.Events.Num();

    std::vector<double> errors;
    std::vector<bool> used(static_cast<size_t>(track.Events.Num()), false);
    int last_match = -1;

    for (const auto& label : handmade)
    {
        const double center = (label.start + label.end) * 0.5;
        int best = -1;
        double best_err = 1e9;
        for (int32 i = 0; i < track.Events.Num(); ++i)
        {
            if (used[static_cast<size_t>(i)] || to_std(track.Events[i].PoseID) != label.pose) continue;
            const double err = std::abs(static_cast<double>(track.Events[i].FinalRenderCenterSeconds) - center);
            if (err < best_err)
            {
                best_err = err;
                best = i;
            }
        }

        if (best >= 0 && best_err <= 0.180)
        {
            used[static_cast<size_t>(best)] = true;
            ++report.matched_count;
            errors.push_back(best_err * 1000.0);
            if (best < last_match) ++report.order_violations;
            last_match = best;
            if (track.Events[best].RenderStartSeconds + 0.050f < label.start) report.early_start_rate += 1.0;
            if (track.Events[best].RenderEndSeconds - 0.050f > label.end) report.late_tail_rate += 1.0;
        }
    }

    report.missing_count = report.reference_count - report.matched_count;
    report.extra_count = report.committed_count - report.matched_count;
    if (!errors.empty())
    {
        double sum = 0.0;
        for (double err : errors) sum += err;
        report.mean_abs_center_error_ms = sum / static_cast<double>(errors.size());
        std::sort(errors.begin(), errors.end());
        const size_t p90_index = std::min(errors.size() - 1, static_cast<size_t>(std::floor(errors.size() * 0.9)));
        report.p90_abs_center_error_ms = errors[p90_index];
        report.early_start_rate /= static_cast<double>(errors.size());
        report.late_tail_rate /= static_cast<double>(errors.size());
    }
    return report;
}

static std::string grade_json(const GradeReport& grade_report)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(3);
    out << "{\n"
        << "  \"reference_count\": " << grade_report.reference_count << ",\n"
        << "  \"committed_count\": " << grade_report.committed_count << ",\n"
        << "  \"matched_count\": " << grade_report.matched_count << ",\n"
        << "  \"missing_count\": " << grade_report.missing_count << ",\n"
        << "  \"extra_count\": " << grade_report.extra_count << ",\n"
        << "  \"order_violations\": " << grade_report.order_violations << ",\n"
        << "  \"mean_abs_center_error_ms\": " << grade_report.mean_abs_center_error_ms << ",\n"
        << "  \"p90_abs_center_error_ms\": " << grade_report.p90_abs_center_error_ms << ",\n"
        << "  \"early_start_rate\": " << grade_report.early_start_rate << ",\n"
        << "  \"late_tail_rate\": " << grade_report.late_tail_rate << "\n"
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
        fs::path root = argc > 1 ? argv[1] : fs::current_path();
        const StreamConfig stream = parse_stream_config(argc, argv);
        fs::path out_root = root / "outputs" / "runs" / "latest";
        fs::remove_all(out_root);
        fs::create_directories(out_root);

        int cases = 0;
        bool all_ok = true;
        for (auto& entry : fs::directory_iterator(root / "inputs" / "transcripts"))
        {
            if (!entry.is_regular_file() || entry.path().extension() != ".txt") continue;

            const std::string stem = entry.path().stem().string();
            const fs::path wav_path = root / "inputs" / "wav" / (stem + ".wav");
            if (!fs::exists(wav_path))
            {
                std::cerr << "missing wav for " << stem << "\n";
                all_ok = false;
                continue;
            }

            const std::string transcript = read_text(entry.path());
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

            const auto handmade = read_handmade_csv(root / "inputs" / "handmade" / (stem + ".csv"));
            const GradeReport report = grade(committed, handmade);
            write_text(case_dir / "grade.json", grade_json(report));

            std::cout << stem
                      << ": planned=" << plan.Events.Num()
                      << " committed=" << committed.Events.Num()
                      << " handmade=" << handmade.size()
                      << " mean_ms=" << report.mean_abs_center_error_ms << "\n";
            ++cases;
        }

        if (cases == 0)
        {
            std::cerr << "no transcript cases found\n";
            return 2;
        }
        return all_ok ? 0 : 1;
    }
    catch (const std::exception& error)
    {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
