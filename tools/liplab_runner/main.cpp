#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "Lipsync/OffgridAITextVisemePlanner.h"
#include "Lipsync/OffgridAIArticulationBudgeter.h"
#include "Lipsync/OffgridAIStreamingSpeechDetector.h"
#include "Lipsync/OffgridAIAudioOccupancyScheduler.h"
#include "Lipsync/OffgridAIVisemePerformer.h"
#include "Lipsync/OffgridAILipsyncRuntimeAdapter.h"
#include "Lipsync/OffgridAIStreamingLandmarkDetector.h"

namespace fs = std::filesystem;

namespace
{
struct FArgs
{
    fs::path InputsDir = "inputs";
    fs::path ConfigPath = "configs/baseline_v26.json";
    fs::path OutputDir = "experiments/runs/local_run";
    int ChunkMs = 40;
    int Fps = 60;
    float PrerollSec = 0.150f;
    float IslandDurationScale = 1.0f;
    bool bWriteDebugCsv = true;
    bool bSimulateRealtime = true;
    int ExpectedSampleRateHz = 0;
    bool bChunkMsOverridden = false;
    bool bFpsOverridden = false;
    bool bPrerollOverridden = false;
};

struct FWavData
{
    int SampleRate = 0;
    int Channels = 0;
    std::vector<int16_t> SamplesInterleaved;
};

struct FTimingDiagnosticRow
{
    int EventIndex = INDEX_NONE;
    std::string PoseID;
    std::string SourceWord;
    int WordIndex = INDEX_NONE;
    int PhraseIndex = 0;
    float RawTextStartSec = 0.0f;
    float RawTextCenterSec = 0.0f;
    float RawTextEndSec = 0.0f;
    float StreamingCenterSec = 0.0f;
    float EvidenceStartSec = 0.0f;
    float EvidenceCenterSec = 0.0f;
    float EvidenceEndSec = 0.0f;
    float RawTextErrorMs = 0.0f;
    float StreamingErrorMs = 0.0f;
    float RawTextLeadBiasMs = 0.0f;
    float StreamingLeadBiasMs = 0.0f;
    float EvidenceConfidence = 0.0f;
    bool bEvidenceFallback = false;
    std::string LandmarkClass;
    bool bNudgeProposed = false;
    bool bNudgeAccepted = false;
    float NudgeDeltaMs = 0.0f;
    std::string NudgeReason;
};

struct FTimingDiagnostics
{
    int MatchedEventCount = 0;
    int EvidenceAnchoredEventCount = 0;
    int AnchoredMatchedEventCount = 0;
    double RawTextAbsErrorSumMs = 0.0;
    double StreamingAbsErrorSumMs = 0.0;
    double AnchoredRawTextAbsErrorSumMs = 0.0;
    double AnchoredStreamingAbsErrorSumMs = 0.0;
    double RawTextLeadBiasSumMs = 0.0;
    double StreamingLeadBiasSumMs = 0.0;
    std::vector<float> RawTextAbsErrorsMs;
    std::vector<float> StreamingAbsErrorsMs;
    std::vector<float> AnchoredRawTextAbsErrorsMs;
    std::vector<float> AnchoredStreamingAbsErrorsMs;
    std::vector<FTimingDiagnosticRow> Rows;
};

static bool IsPose(FName PoseID, const TCHAR* Literal)
{
    return PoseID == FName(Literal);
}

static bool IsClosurePose(FName PoseID)
{
    return IsPose(PoseID, TEXT("22_MBP"));
}

static bool IsExplosivePose(FName PoseID)
{
    return IsClosurePose(PoseID)
        || IsPose(PoseID, TEXT("20_FV"))
        || IsPose(PoseID, TEXT("19_FV-Or-"))
        || IsPose(PoseID, TEXT("21_FV-Ee-"))
        || IsPose(PoseID, TEXT("14_ChJjSh"));
}

static bool IsGlidePose(FName PoseID)
{
    return IsPose(PoseID, TEXT("12_Ww-Oo-")) || IsPose(PoseID, TEXT("16_Ww-Ew-"));
}

static bool IsRoundOrWideVowelPose(FName PoseID)
{
    return IsGlidePose(PoseID) || IsPose(PoseID, TEXT("11_Oo"))
        || IsPose(PoseID, TEXT("10_Or")) || IsPose(PoseID, TEXT("03_Ee"))
        || IsPose(PoseID, TEXT("05_Ay")) || IsPose(PoseID, TEXT("06_Eh"))
        || IsPose(PoseID, TEXT("07_Aa")) || IsPose(PoseID, TEXT("08_Ah"))
        || IsPose(PoseID, TEXT("09_Oh")) || IsPose(PoseID, TEXT("18_Uh"));
}

static bool IsVowelLikePose(FName PoseID)
{
    return IsRoundOrWideVowelPose(PoseID) || IsPose(PoseID, TEXT("04_Ih"));
}

static float DesiredPeakHoldHalfSecondsForPose(FName PoseID)
{
    if (IsClosurePose(PoseID)) { return 0.034f; }
    if (IsPose(PoseID, TEXT("20_FV")) || IsPose(PoseID, TEXT("19_FV-Or-")) || IsPose(PoseID, TEXT("21_FV-Ee-"))) { return 0.028f; }
    if (IsPose(PoseID, TEXT("14_ChJjSh"))) { return 0.024f; }
    if (IsPose(PoseID, TEXT("11_Oo")) || IsPose(PoseID, TEXT("12_Ww-Oo-")) || IsPose(PoseID, TEXT("16_Ww-Ew-")) || IsPose(PoseID, TEXT("10_Or"))) { return 0.020f; }
    if (IsRoundOrWideVowelPose(PoseID)) { return 0.014f; }
    if (IsPose(PoseID, TEXT("24_Tongue_Th"))) { return 0.010f; }
    return 0.012f;
}

static std::string ReadTextFile(const fs::path& Path)
{
    std::ifstream In(Path, std::ios::binary);
    if (!In)
    {
        return {};
    }
    std::ostringstream SS;
    SS << In.rdbuf();
    std::string S = SS.str();
    while (!S.empty() && (S.back() == '\r' || S.back() == '\n' || S.back() == '\0'))
    {
        S.pop_back();
    }
    return S;
}

static std::string ReadJsonStringFile(const fs::path& Path, std::string& Error)
{
    std::ifstream In(Path, std::ios::binary);
    if (!In)
    {
        Error = "could not open config";
        return {};
    }
    std::ostringstream SS;
    SS << In.rdbuf();
    return SS.str();
}

static std::optional<int> TryExtractInt(const std::string& Json, const std::string& Key)
{
    const std::regex Pattern("\"" + Key + "\"\\s*:\\s*(-?\\d+)");
    std::smatch Match;
    if (!std::regex_search(Json, Match, Pattern) || Match.size() < 2)
    {
        return std::nullopt;
    }
    try
    {
        return std::stoi(Match[1].str());
    }
    catch (const std::exception&)
    {
        return std::nullopt;
    }
}

static std::optional<float> TryExtractFloat(const std::string& Json, const std::string& Key)
{
    const std::regex Pattern("\"" + Key + "\"\\s*:\\s*(-?\\d+(?:\\.\\d+)?)");
    std::smatch Match;
    if (!std::regex_search(Json, Match, Pattern) || Match.size() < 2)
    {
        return std::nullopt;
    }
    try
    {
        return std::stof(Match[1].str());
    }
    catch (const std::exception&)
    {
        return std::nullopt;
    }
}

static std::optional<bool> TryExtractBool(const std::string& Json, const std::string& Key)
{
    const std::regex Pattern("\"" + Key + "\"\\s*:\\s*(true|false)", std::regex::icase);
    std::smatch Match;
    if (!std::regex_search(Json, Match, Pattern) || Match.size() < 2)
    {
        return std::nullopt;
    }
    const std::string Value = Match[1].str();
    if (Value == "true" || Value == "TRUE" || Value == "True")
    {
        return true;
    }
    if (Value == "false" || Value == "FALSE" || Value == "False")
    {
        return false;
    }
    return std::nullopt;
}

static bool LoadConfig(const fs::path& ConfigPath, FArgs& Args, std::string& Error)
{
    if (ConfigPath.empty())
    {
        return true;
    }
    const std::string Json = ReadJsonStringFile(ConfigPath, Error);
    if (!Error.empty())
    {
        return false;
    }

    if (!Args.bPrerollOverridden)
    {
        if (const auto PrerollMs = TryExtractFloat(Json, "preroll_ms"))
        {
            Args.PrerollSec = std::max(0.0f, *PrerollMs / 1000.0f);
        }
    }
    if (!Args.bChunkMsOverridden)
    {
        if (const auto ChunkMs = TryExtractInt(Json, "streaming_chunk_ms"))
        {
            Args.ChunkMs = std::max(1, *ChunkMs);
        }
    }
    if (const auto WriteDebugCsv = TryExtractBool(Json, "write_debug_csv"))
    {
        Args.bWriteDebugCsv = *WriteDebugCsv;
    }
    if (const auto SimulateRealtime = TryExtractBool(Json, "simulate_realtime"))
    {
        Args.bSimulateRealtime = *SimulateRealtime;
    }
    if (const auto SampleRateHz = TryExtractInt(Json, "sample_rate_hz"))
    {
        Args.ExpectedSampleRateHz = std::max(0, *SampleRateHz);
    }
    if (const auto RequireAllCommitted = TryExtractBool(Json, "require_all_budgeted_events_committed"))
    {
        if (!*RequireAllCommitted)
        {
            Error = "config disables require_all_budgeted_events_committed, which liplab_runner does not support";
            return false;
        }
    }
    if (const auto RequireMonotonic = TryExtractBool(Json, "require_monotonic_committed_centers"))
    {
        if (!*RequireMonotonic)
        {
            Error = "config disables require_monotonic_committed_centers, which liplab_runner does not support";
            return false;
        }
    }
    if (!Args.bSimulateRealtime)
    {
        Error = "simulate_realtime=false is not supported by the current offline runner";
        return false;
    }
    return true;
}

static uint32_t ReadU32(std::ifstream& In)
{
    uint8_t B[4]{};
    In.read(reinterpret_cast<char*>(B), 4);
    return uint32_t(B[0]) | (uint32_t(B[1]) << 8) | (uint32_t(B[2]) << 16) | (uint32_t(B[3]) << 24);
}

static uint16_t ReadU16(std::ifstream& In)
{
    uint8_t B[2]{};
    In.read(reinterpret_cast<char*>(B), 2);
    return uint16_t(B[0]) | (uint16_t(B[1]) << 8);
}

static bool ReadWavPCM16(const fs::path& Path, FWavData& Out, std::string& Error)
{
    std::ifstream In(Path, std::ios::binary);
    if (!In)
    {
        Error = "could not open WAV";
        return false;
    }

    char Riff[4]{};
    In.read(Riff, 4);
    (void)ReadU32(In);
    char Wave[4]{};
    In.read(Wave, 4);
    if (std::string(Riff, 4) != "RIFF" || std::string(Wave, 4) != "WAVE")
    {
        Error = "not a RIFF/WAVE file";
        return false;
    }

    bool bHaveFmt = false;
    bool bHaveData = false;
    uint16_t AudioFormat = 0;
    uint16_t NumChannels = 0;
    uint32_t SampleRate = 0;
    uint16_t BitsPerSample = 0;
    std::vector<uint8_t> DataBytes;

    while (In && (!bHaveFmt || !bHaveData))
    {
        char ChunkIdChars[4]{};
        In.read(ChunkIdChars, 4);
        if (!In) { break; }
        const std::string ChunkId(ChunkIdChars, 4);
        const uint32_t ChunkSize = ReadU32(In);
        const std::streampos ChunkStart = In.tellg();

        if (ChunkId == "fmt ")
        {
            AudioFormat = ReadU16(In);
            NumChannels = ReadU16(In);
            SampleRate = ReadU32(In);
            (void)ReadU32(In); // byte rate
            (void)ReadU16(In); // block align
            BitsPerSample = ReadU16(In);
            bHaveFmt = true;
        }
        else if (ChunkId == "data")
        {
            DataBytes.resize(ChunkSize);
            In.read(reinterpret_cast<char*>(DataBytes.data()), ChunkSize);
            bHaveData = true;
        }

        In.seekg(ChunkStart + static_cast<std::streamoff>(ChunkSize + (ChunkSize & 1u)));
    }

    if (!bHaveFmt || !bHaveData)
    {
        Error = "missing fmt or data chunk";
        return false;
    }
    if (AudioFormat != 1 || BitsPerSample != 16 || NumChannels == 0 || SampleRate == 0)
    {
        std::ostringstream SS;
        SS << "unsupported WAV format: format=" << AudioFormat << " bits=" << BitsPerSample << " channels=" << NumChannels;
        Error = SS.str();
        return false;
    }

    const size_t SampleCount = DataBytes.size() / sizeof(int16_t);
    Out.SamplesInterleaved.resize(SampleCount);
    for (size_t I = 0; I < SampleCount; ++I)
    {
        const uint8_t Lo = DataBytes[I * 2 + 0];
        const uint8_t Hi = DataBytes[I * 2 + 1];
        Out.SamplesInterleaved[I] = static_cast<int16_t>(uint16_t(Lo) | (uint16_t(Hi) << 8));
    }
    Out.SampleRate = static_cast<int>(SampleRate);
    Out.Channels = static_cast<int>(NumChannels);
    return true;
}

static std::string CsvEscape(const std::string& S)
{
    bool bNeedsQuotes = false;
    for (char C : S)
    {
        if (C == ',' || C == '"' || C == '\n' || C == '\r')
        {
            bNeedsQuotes = true;
            break;
        }
    }
    if (!bNeedsQuotes) { return S; }
    std::string R = "\"";
    for (char C : S)
    {
        if (C == '"') { R += "\"\""; }
        else { R.push_back(C); }
    }
    R += "\"";
    return R;
}

static std::string VisemeToString(EOffgridAITextViseme V)
{
    return FOffgridAITextVisemePlanner::ToDebugString(V).ToStdString();
}


static bool IsNudgeEligiblePose(FName PoseID)
{
    return IsClosurePose(PoseID)
        || IsPose(PoseID, TEXT("20_FV"))
        || IsPose(PoseID, TEXT("19_FV-Or-"))
        || IsPose(PoseID, TEXT("21_FV-Ee-"))
        || IsPose(PoseID, TEXT("14_ChJjSh"))
        || IsPose(PoseID, TEXT("12_Ww-Oo-"))
        || IsPose(PoseID, TEXT("16_Ww-Ew-"))
        || IsPose(PoseID, TEXT("11_Oo"))
        || IsPose(PoseID, TEXT("10_Or"))
        || IsPose(PoseID, TEXT("03_Ee"))
        || IsPose(PoseID, TEXT("07_Aa"))
        || IsPose(PoseID, TEXT("08_Ah"));
}

static std::string LandmarkClassForPose(FName PoseID)
{
    if (IsClosurePose(PoseID)) { return "MBP"; }
    if (IsPose(PoseID, TEXT("20_FV")) || IsPose(PoseID, TEXT("19_FV-Or-")) || IsPose(PoseID, TEXT("21_FV-Ee-"))) { return "FV"; }
    if (IsPose(PoseID, TEXT("12_Ww-Oo-")) || IsPose(PoseID, TEXT("16_Ww-Ew-"))) { return "WOO_GLIDE"; }
    if (IsPose(PoseID, TEXT("11_Oo")) || IsPose(PoseID, TEXT("10_Or"))) { return "ROUNDED_VOWEL"; }
    if (IsPose(PoseID, TEXT("14_ChJjSh"))) { return "SH_CH_J"; }
    if (IsPose(PoseID, TEXT("03_Ee")) || IsPose(PoseID, TEXT("05_Ay")) || IsPose(PoseID, TEXT("06_Eh")) || IsPose(PoseID, TEXT("04_Ih"))) { return "FRONT_VOWEL"; }
    if (IsPose(PoseID, TEXT("07_Aa")) || IsPose(PoseID, TEXT("08_Ah")) || IsPose(PoseID, TEXT("09_Oh")) || IsPose(PoseID, TEXT("18_Uh"))) { return "OPEN_VOWEL"; }
    if (IsPose(PoseID, TEXT("24_Tongue_Th"))) { return "TH_WEAK"; }
    if (IsVowelLikePose(PoseID)) { return "VOWEL_OTHER"; }
    return "OTHER";
}

static std::string NudgeReasonForEvent(const FOffgridAIAlignedVisemeEvent& E)
{
    const bool bEligible = IsNudgeEligiblePose(E.PoseID);
    const float AbsShiftMs = FMath::Abs(E.AppliedShiftSeconds) * 1000.0f;
    if (!bEligible) { return "not_nudge_eligible_pose"; }
    if (E.AudioNudgeRejectReason != NAME_None) { return E.AudioNudgeRejectReason.ToString().ToStdString(); }
    if (E.AlignmentConfidence <= 0.0f) { return "no_audio_evidence"; }
    if (AbsShiftMs < 0.5f) { return "evidence_no_shift"; }
    return "accepted_local_nudge";
}

static bool WasNudgeProposedForEvent(const FOffgridAIAlignedVisemeEvent& E)
{
    return IsNudgeEligiblePose(E.PoseID);
}

static bool WasNudgeAcceptedForEvent(const FOffgridAIAlignedVisemeEvent& E)
{
    const float Confidence = FMath::Max(E.AlignmentConfidence, E.AudioNudgeCandidateConfidence);
    return WasNudgeProposedForEvent(E) && E.bAudioNudgeAccepted && Confidence > 0.0f && FMath::Abs(E.AppliedShiftSeconds) > 0.0005f;
}


static bool IsAlpha31PrimaryAuditPose(FName PoseID)
{
    return IsClosurePose(PoseID)
        || IsPose(PoseID, TEXT("20_FV"))
        || IsPose(PoseID, TEXT("19_FV-Or-"))
        || IsPose(PoseID, TEXT("21_FV-Ee-"))
        || IsPose(PoseID, TEXT("12_Ww-Oo-"))
        || IsPose(PoseID, TEXT("16_Ww-Ew-"))
        || IsPose(PoseID, TEXT("11_Oo"))
        || IsPose(PoseID, TEXT("10_Or"));
}


static bool IsAu23AnchorFrontierPose(FName PoseID)
{
    const std::string Class = LandmarkClassForPose(PoseID);
    return Class == "MBP" ||
        Class == "FV" ||
        Class == "WOO_GLIDE" ||
        Class == "ROUNDED_VOWEL" ||
        Class == "SH_CH_J" ||
        Class == "FRONT_VOWEL" ||
        Class == "OPEN_VOWEL" ||
        Class == "TH_WEAK";
}

static float Au23RequiredAnchorConfidenceForClass(const std::string& Class)
{
    if (Class == "MBP" || Class == "FV") { return 0.09f; }
    if (Class == "WOO_GLIDE" || Class == "ROUNDED_VOWEL" || Class == "SH_CH_J") { return 0.14f; }
    if (Class == "FRONT_VOWEL" || Class == "OPEN_VOWEL") { return 0.22f; }
    return 0.30f;
}

static bool IsAu23CurrentHardAnchorClass(const std::string& Class)
{
    return Class == "MBP" || Class == "FV" || Class == "WOO_GLIDE" || Class == "SH_CH_J";
}

static bool IsAu24SoftTimelineAnchorClass(const std::string& Class)
{
    return Class == "ROUNDED_VOWEL" || Class == "FRONT_VOWEL";
}

static bool IsAu24TimelineAnchorClass(const std::string& Class)
{
    return Class == "MBP" || Class == "FV" || Class == "WOO_GLIDE" || IsAu24SoftTimelineAnchorClass(Class);
}

static float Au24CalibratedTimelineAnchorConfidenceForClass(const std::string& Class)
{
    // AU24b/AU25: corpus-calibrated timeline-anchor gates.  MBP/FV remain
    // strong hard anchors.  WOO is useful but less reliable.  Rounded/front
    // vowels are admitted only as soft timeline anchors, not as hard visual
    // event nudges.  SH/CH and TH remain diagnostic-only after AU23a.
    if (Class == "MBP") { return 0.12f; }
    if (Class == "FV") { return 0.12f; }
    if (Class == "WOO_GLIDE") { return 0.18f; }
    if (Class == "ROUNDED_VOWEL") { return 0.22f; }
    if (Class == "FRONT_VOWEL") { return 0.26f; }
    return 1.01f;
}

static bool IsAu24SoftTimelineAnchorPose(FName PoseID)
{
    return IsAu24SoftTimelineAnchorClass(LandmarkClassForPose(PoseID));
}

static bool IsAu24TimelineAnchorPose(FName PoseID)
{
    return IsAu24TimelineAnchorClass(LandmarkClassForPose(PoseID));
}

static float Alpha31ClampMinMs(FName PoseID)
{
    if (IsClosurePose(PoseID)) { return -80.0f; }
    if (IsPose(PoseID, TEXT("20_FV")) || IsPose(PoseID, TEXT("19_FV-Or-")) || IsPose(PoseID, TEXT("21_FV-Ee-"))) { return -60.0f; }
    if (IsPose(PoseID, TEXT("12_Ww-Oo-")) || IsPose(PoseID, TEXT("16_Ww-Ew-")) || IsPose(PoseID, TEXT("11_Oo")) || IsPose(PoseID, TEXT("10_Or"))) { return -70.0f; }
    return -50.0f;
}

static float Alpha31ClampMaxMs(FName PoseID)
{
    if (IsClosurePose(PoseID)) { return 30.0f; }
    if (IsPose(PoseID, TEXT("20_FV")) || IsPose(PoseID, TEXT("19_FV-Or-")) || IsPose(PoseID, TEXT("21_FV-Ee-"))) { return 40.0f; }
    if (IsPose(PoseID, TEXT("12_Ww-Oo-")) || IsPose(PoseID, TEXT("16_Ww-Ew-")) || IsPose(PoseID, TEXT("11_Oo")) || IsPose(PoseID, TEXT("10_Or"))) { return 70.0f; }
    return 50.0f;
}

static float Alpha31VisualLeadSeconds(FName PoseID)
{
    if (IsClosurePose(PoseID)) { return 0.045f; }
    if (IsPose(PoseID, TEXT("20_FV")) || IsPose(PoseID, TEXT("19_FV-Or-")) || IsPose(PoseID, TEXT("21_FV-Ee-"))) { return 0.025f; }
    return 0.0f;
}

static std::string Alpha31RejectReason(
    bool bHasEvidence,
    bool bStrongEvidence,
    bool bWouldCrossPrev,
    bool bWouldCrossNext,
    bool bWouldLeaveIsland,
    bool bExceedsClamp,
    float AbsClampedShiftMs)
{
    if (!bHasEvidence) { return "no_offline_audio_evidence"; }
    if (!bStrongEvidence) { return "low_offline_audio_confidence"; }
    if (bWouldCrossPrev) { return "would_cross_prev_event"; }
    if (bWouldCrossNext) { return "would_cross_next_event"; }
    if (bWouldLeaveIsland) { return "would_leave_island_bounds"; }
    if (bExceedsClamp) { return "would_exceed_class_clamp"; }
    if (AbsClampedShiftMs < 0.5f) { return "evidence_no_shift"; }
    return "hypothetically_safe";
}

static TArray<int16> BuildMonoPCM16(const FWavData& Wav)
{
    TArray<int16> Mono;
    if (Wav.Channels <= 0 || Wav.SamplesInterleaved.empty())
    {
        return Mono;
    }

    const int TotalFrames = static_cast<int>(Wav.SamplesInterleaved.size()) / Wav.Channels;
    Mono.Reserve(TotalFrames);
    for (int FrameIndex = 0; FrameIndex < TotalFrames; ++FrameIndex)
    {
        int Sum = 0;
        for (int Channel = 0; Channel < Wav.Channels; ++Channel)
        {
            Sum += static_cast<int>(Wav.SamplesInterleaved[FrameIndex * Wav.Channels + Channel]);
        }
        Mono.Add(static_cast<int16>(Sum / Wav.Channels));
    }
    return Mono;
}



static void WriteStreamTailDiagnosticsCsv(const fs::path& Path, const FOffgridAIStreamTailDiagnosticRow& R)
{
    std::ofstream Out(Path);
    Out << "LineID,PCMChunkCount,PCMBytesReceived,PCMSamplesReceived,LastSampleRate,LastNumChannels,LastChunkStartSample,LastChunkEndSample,ObservedAudioBufferEndSec,FirstSpeechAudioBufferStartSec,SpeechIslandCount,InputStreamClosed,DiagnosticKind\n";
    Out << CsvEscape(R.LineID.ToString().ToStdString()) << ','
        << R.PCMChunkCount << ','
        << R.PCMBytesReceived << ','
        << R.PCMSamplesReceived << ','
        << R.LastSampleRate << ','
        << R.LastNumChannels << ','
        << R.LastChunkStartSample << ','
        << R.LastChunkEndSample << ','
        << R.ObservedAudioBufferEndSec << ','
        << R.FirstSpeechAudioBufferStartSec << ','
        << R.SpeechIslandCount << ','
        << (R.bInputStreamClosed ? 1 : 0) << ','
        << CsvEscape(R.DiagnosticKind.ToString().ToStdString()) << "\n";
}

static void WriteAudioOccupancyDiagnosticsCsv(const fs::path& Path, const TArray<FOffgridAIAudioOccupancyDiagnosticRow>& Rows)
{
    std::ofstream Out(Path);
    Out << "LineID,UpdateOrdinal,FinalReplay,CurrentPlaybackSec,PrerollSec,SourceEventIndex,Word,PoseID,PlannedCenterSec,CommittedCenterSec,RenderStartSec,RenderEndSec,CommitReason,PlaybackMode,SpeechIslandIndex,SpeechIslandStartSec,SpeechIslandEndSec,AudioActiveSec,TextPlayheadSec,TailDrain,DiagnosticKind\n";
    for (const FOffgridAIAudioOccupancyDiagnosticRow& R : Rows)
    {
        Out << CsvEscape(R.LineID.ToString().ToStdString()) << ','
            << R.UpdateOrdinal << ','
            << (R.bFinalReplay ? 1 : 0) << ','
            << R.CurrentPlaybackSec << ','
            << R.PrerollSec << ','
            << R.SourceEventIndex << ','
            << CsvEscape(R.Word.ToStdString()) << ','
            << CsvEscape(R.PoseID.ToString().ToStdString()) << ','
            << R.PlannedCenterSec << ','
            << R.CommittedCenterSec << ','
            << R.RenderStartSec << ','
            << R.RenderEndSec << ','
            << CsvEscape(R.CommitReason.ToString().ToStdString()) << ','
            << CsvEscape(R.PlaybackMode.ToString().ToStdString()) << ','
            << R.SpeechIslandIndex << ','
            << R.SpeechIslandStartSec << ','
            << R.SpeechIslandEndSec << ','
            << R.AudioActiveSec << ','
            << R.TextPlayheadSec << ','
            << (R.bTailDrain ? 1 : 0) << ','
            << CsvEscape(R.DiagnosticKind.ToString().ToStdString()) << "\n";
    }
}



static void WritePlannedEventsCsv(const fs::path& Path, const FOffgridAITextVisemePlan& Plan)
{
    std::ofstream Out(Path);
    Out << "EventIndex,PoseID,Viseme,SourceWord,WordIndex,PhraseIndex,SentenceIslandIndex,StartNorm,EndNorm,TextCenterSec,Strength,Generator\n";
    for (int32 I = 0; I < Plan.Events.Num(); ++I)
    {
        const auto& E = Plan.Events[I];
        const float TextCenterSec = 0.5f * (E.StartNorm + E.EndNorm) * Plan.EstimatedDurationSeconds;
        Out << I << ','
            << CsvEscape(E.PoseID.ToString().ToStdString()) << ','
            << CsvEscape(VisemeToString(E.Viseme)) << ','
            << CsvEscape(E.SourceText.ToStdString()) << ','
            << E.WordIndex << ','
            << E.PhraseIndex << ','
            << E.SentenceIslandIndex << ','
            << E.StartNorm << ','
            << E.EndNorm << ','
            << TextCenterSec << ','
            << E.Strength << ','
            << CsvEscape(E.Generator.ToString().ToStdString()) << "\n";
    }
}

static void WriteCommittedEventsCsv(const fs::path& Path, const FOffgridAIAlignedVisemeTrack& Track)
{
    std::ofstream Out(Path);
    Out << "ArrayIndex,EventIndex,PoseID,SourceWord,WordIndex,PhraseIndex,TextIslandIndex,AudioIslandIndex,IslandLocalNorm,IslandAudioStartSec,IslandAudioEndSec,LandmarkClass,TextCenterSec,CommittedPlaybackCenterSec,RenderStartSec,RenderEndSec,Strength,CommitLeadSec,CommitReason,CommitHorizonSec,EffectiveSegmentScale,AlignmentConfidence,RawShiftMs,AppliedShiftMs,CenterOrderRepaired,CenterOrderRepairMs\n";
    for (int32 I = 0; I < Track.Events.Num(); ++I)
    {
        const FOffgridAIAlignedVisemeEvent& E = Track.Events[I];
        Out << I << ','
            << E.EventIndex << ','
            << CsvEscape(E.PoseID.ToString().ToStdString()) << ','
            << CsvEscape(E.SourceWord.ToStdString()) << ','
            << E.WordIndex << ','
            << E.PhraseIndex << ','
            << E.TextIslandIndex << ','
            << E.AudioIslandIndex << ','
            << E.IslandLocalNorm << ','
            << E.IslandAudioStartSeconds << ','
            << E.IslandAudioEndSeconds << ','
            << CsvEscape(LandmarkClassForPose(E.PoseID)) << ','
            << E.TextDiagnosticCenterSeconds << ','
            << E.FinalRenderCenterSeconds << ','
            << E.RenderStartSeconds << ','
            << E.RenderEndSeconds << ','
            << E.Strength << ','
            << E.CommitLeadSeconds << ','
            << CsvEscape(E.CommitReason.ToString().ToStdString()) << ','
            << E.EffectiveCommitHorizonSeconds << ','
            << E.EffectiveSegmentScale << ','
            << E.AlignmentConfidence << ','
            << (E.RawShiftSeconds * 1000.0f) << ','
            << (E.AppliedShiftSeconds * 1000.0f) << ','
            << (E.bCenterOrderRepaired ? 1 : 0) << ','
            << (E.CenterOrderRepairSeconds * 1000.0f) << "\n";
    }
}


static void WriteDurationScalingDiagnosticsCsv(
    const fs::path& EventPath,
    const fs::path& SummaryPath,
    const FOffgridAITextVisemePlan& Plan,
    const FOffgridAIAlignedVisemeTrack& Track,
    const TArray<FOffgridAIStreamingSpeechIsland>& AudioIslands)
{
    struct FDiagnosticRegion
    {
        int32 RegionIndex = INDEX_NONE;
        int32 FirstEventIndex = INDEX_NONE;
        int32 LastEventIndex = INDEX_NONE;
        float PlanStartSec = 0.0f;
        float PlanEndSec = 0.0f;
        float AudioStartSec = 0.0f;
        float AudioEndSec = 0.0f;
        bool bHasAudio = false;
    };

    auto PlanEventStartSec = [&](int32 EventIndex, float FallbackCenter) -> float
    {
        if (Plan.Events.IsValidIndex(EventIndex))
        {
            const FOffgridAITextVisemeEvent& PE = Plan.Events[EventIndex];
            return FMath::Clamp(PE.StartNorm, 0.0f, 1.0f) * FMath::Max(Plan.EstimatedDurationSeconds, 0.001f);
        }
        return FallbackCenter;
    };
    auto PlanEventEndSec = [&](int32 EventIndex, float FallbackCenter) -> float
    {
        if (Plan.Events.IsValidIndex(EventIndex))
        {
            const FOffgridAITextVisemeEvent& PE = Plan.Events[EventIndex];
            return FMath::Clamp(PE.EndNorm, PE.StartNorm, 1.0f) * FMath::Max(Plan.EstimatedDurationSeconds, 0.001f);
        }
        return FallbackCenter;
    };
    auto NormalizedWord = [](const FString& In) -> std::string
    {
        std::string S = In.ToStdString();
        std::string Out;
        Out.reserve(S.size());
        for (char C : S)
        {
            if (std::isalnum(static_cast<unsigned char>(C)) || C == '\'')
            {
                Out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(C))));
            }
        }
        return Out;
    };
    auto IsSoftBoundaryWord = [&](const FString& In) -> bool
    {
        const std::string W = NormalizedWord(In);
        static const std::set<std::string> BoundaryWords = {
            "and", "but", "or", "then", "so", "because", "with", "for", "to", "from", "while", "after", "before", "would", "could", "can", "let"
        };
        return BoundaryWords.find(W) != BoundaryWords.end();
    };
    auto EventAudioEnd = [](const FOffgridAIAlignedVisemeEvent& E) -> float
    {
        return E.IslandAudioSpanSeconds > 0.0f
            ? E.IslandAudioStartSeconds + E.IslandAudioSpanSeconds
            : FMath::Max(E.IslandAudioEndSeconds, E.IslandAudioStartSeconds);
    };

    std::vector<FDiagnosticRegion> SyntheticRegions;
    {
        const int32 MinEventsPerRegion = 5;
        const int32 TargetEventsPerRegion = 10;
        const int32 MaxEventsPerRegion = 12;
        const float MaxRegionPlanSec = 1.25f;

        int32 RegionStartArrayIndex = 0;
        while (RegionStartArrayIndex < Track.Events.Num())
        {
            int32 RegionEndArrayIndex = RegionStartArrayIndex;
            int32 LastWordIndex = Track.Events[RegionStartArrayIndex].WordIndex;
            for (int32 I = RegionStartArrayIndex + 1; I < Track.Events.Num(); ++I)
            {
                const int32 EventCount = I - RegionStartArrayIndex + 1;
                const float RegionPlanSpan = PlanEventEndSec(Track.Events[I].EventIndex, Track.Events[I].TextDiagnosticCenterSeconds)
                    - PlanEventStartSec(Track.Events[RegionStartArrayIndex].EventIndex, Track.Events[RegionStartArrayIndex].TextDiagnosticCenterSeconds);
                const bool bWordAdvanced = Track.Events[I].WordIndex != LastWordIndex;
                const bool bSoftBoundary = bWordAdvanced && IsSoftBoundaryWord(Track.Events[I].SourceWord);
                const bool bHitTargetAtWordBoundary = EventCount >= TargetEventsPerRegion && bWordAdvanced;
                const bool bHitSoftBoundary = EventCount >= MinEventsPerRegion && bSoftBoundary;
                const bool bHitMaxCount = EventCount >= MaxEventsPerRegion;
                const bool bHitMaxDuration = EventCount >= MinEventsPerRegion && RegionPlanSpan >= MaxRegionPlanSec && bWordAdvanced;
                if (bHitSoftBoundary || bHitTargetAtWordBoundary || bHitMaxCount || bHitMaxDuration)
                {
                    break;
                }
                RegionEndArrayIndex = I;
                LastWordIndex = Track.Events[I].WordIndex;
            }

            FDiagnosticRegion R;
            R.RegionIndex = static_cast<int32>(SyntheticRegions.size());
            R.FirstEventIndex = Track.Events[RegionStartArrayIndex].EventIndex;
            R.LastEventIndex = Track.Events[RegionEndArrayIndex].EventIndex;
            R.PlanStartSec = PlanEventStartSec(Track.Events[RegionStartArrayIndex].EventIndex, Track.Events[RegionStartArrayIndex].TextDiagnosticCenterSeconds);
            R.PlanEndSec = PlanEventEndSec(Track.Events[RegionEndArrayIndex].EventIndex, Track.Events[RegionEndArrayIndex].TextDiagnosticCenterSeconds);
            R.AudioStartSec = std::numeric_limits<float>::max();
            R.AudioEndSec = 0.0f;
            for (int32 I = RegionStartArrayIndex; I <= RegionEndArrayIndex; ++I)
            {
                const FOffgridAIAlignedVisemeEvent& E = Track.Events[I];
                if (E.AudioIslandIndex != INDEX_NONE || E.IslandAudioSpanSeconds > 0.0f)
                {
                    const float AS = E.IslandAudioStartSeconds;
                    const float AE = EventAudioEnd(E);
                    if (AE > AS)
                    {
                        R.AudioStartSec = FMath::Min(R.AudioStartSec, AS);
                        R.AudioEndSec = FMath::Max(R.AudioEndSec, AE);
                        R.bHasAudio = true;
                    }
                }
            }
            if (!R.bHasAudio)
            {
                R.AudioStartSec = 0.0f;
                R.AudioEndSec = 0.0f;
            }
            SyntheticRegions.push_back(R);
            RegionStartArrayIndex = RegionEndArrayIndex + 1;
        }
    }

    std::map<int32, const FDiagnosticRegion*> SyntheticByEventIndex;
    for (const FDiagnosticRegion& R : SyntheticRegions)
    {
        for (const FOffgridAIAlignedVisemeEvent& E : Track.Events)
        {
            if (E.EventIndex >= R.FirstEventIndex && E.EventIndex <= R.LastEventIndex)
            {
                SyntheticByEventIndex[E.EventIndex] = &R;
            }
        }
    }

    float ObservedActiveSec = 0.0f;
    float FirstSpeechStartSec = 0.0f;
    bool bHasSpeech = false;
    for (const FOffgridAIStreamingSpeechIsland& Island : AudioIslands)
    {
        const float Start = Island.AudioBufferStartSec;
        const float End = FMath::Max(Island.AudioBufferLastSpeechSec, Island.AudioBufferEndSec);
        if (End > Start)
        {
            if (!bHasSpeech)
            {
                FirstSpeechStartSec = Start;
                bHasSpeech = true;
            }
            ObservedActiveSec += End - Start;
        }
    }

    const float PlannedActiveSec = FMath::Max(Plan.EstimatedDurationSeconds, 0.001f);
    const float GlobalScale = bHasSpeech ? (ObservedActiveSec / PlannedActiveSec) : 1.0f;
    int32 TailFlushCount = 0;
    int32 FallbackCount = 0;
    for (const FOffgridAIAlignedVisemeEvent& E : Track.Events)
    {
        const std::string Reason = E.CommitReason.ToString().ToStdString();
        if (Reason.find("flush") != std::string::npos)
        {
            ++TailFlushCount;
        }
        if (E.AudioIslandIndex == INDEX_NONE || Reason.find("fallback") != std::string::npos || Reason.find("missing") != std::string::npos)
        {
            ++FallbackCount;
        }
    }

    std::ofstream Out(EventPath);
    Out << "EventIndex,PoseID,SourceWord,WordIndex,TextIslandIndex,AudioIslandIndex,SyntheticRegionIndex,SyntheticRegionFirstEventIndex,SyntheticRegionLastEventIndex,PlanTextCenterSec,ActualCommittedCenterSec,GlobalScaledCounterfactualCenterSec,IslandScaledCounterfactualCenterSec,SyntheticRegionScaledCounterfactualCenterSec,ActualMinusPlanMs,ActualMinusGlobalScaledMs,ActualMinusIslandScaledMs,ActualMinusSyntheticRegionScaledMs,ObservedActiveSec,PlannedActiveSec,GlobalScale,IslandObservedSec,IslandPlannedSec,IslandScale,SyntheticRegionObservedSec,SyntheticRegionPlannedSec,SyntheticRegionScale,CommitReason,PlaybackModeHint\n";

    double ActualAbsMs = 0.0;
    double GlobalAbsMs = 0.0;
    double IslandAbsMs = 0.0;
    double SyntheticAbsMs = 0.0;
    int32 ComparedCount = 0;
    int32 SyntheticCoveredEventCount = 0;

    for (const FOffgridAIAlignedVisemeEvent& E : Track.Events)
    {
        const float PlanCenter = E.TextDiagnosticCenterSeconds;
        const float ActualCenter = E.FinalRenderCenterSeconds;
        const float GlobalCenter = bHasSpeech ? (FirstSpeechStartSec + PlanCenter * GlobalScale) : PlanCenter;
        const float IslandObserved = E.IslandAudioSpanSeconds > 0.0f
            ? E.IslandAudioSpanSeconds
            : FMath::Max(0.0f, E.IslandAudioEndSeconds - E.IslandAudioStartSeconds);
        const float IslandPlanned = E.PlannerIslandPredictedDurationSeconds > 0.0f
            ? E.PlannerIslandPredictedDurationSeconds
            : PlannedActiveSec;
        const float IslandScale = IslandPlanned > 0.001f ? IslandObserved / IslandPlanned : 1.0f;
        const float IslandOffset = E.PlannerEventPredictedOffsetSeconds > 0.0f
            ? E.PlannerEventPredictedOffsetSeconds
            : FMath::Max(0.0f, PlanCenter - E.IslandTextStartSeconds);
        const float IslandCenter = (E.IslandAudioStartSeconds > 0.0f || IslandObserved > 0.0f)
            ? (E.IslandAudioStartSeconds + IslandOffset * IslandScale)
            : GlobalCenter;

        int32 SyntheticRegionIndex = INDEX_NONE;
        int32 SyntheticRegionFirstEventIndex = INDEX_NONE;
        int32 SyntheticRegionLastEventIndex = INDEX_NONE;
        float SyntheticObserved = 0.0f;
        float SyntheticPlanned = 0.0f;
        float SyntheticScale = 1.0f;
        float SyntheticCenter = GlobalCenter;
        auto SyntheticIt = SyntheticByEventIndex.find(E.EventIndex);
        if (SyntheticIt != SyntheticByEventIndex.end() && SyntheticIt->second != nullptr)
        {
            const FDiagnosticRegion& R = *SyntheticIt->second;
            SyntheticRegionIndex = R.RegionIndex;
            SyntheticRegionFirstEventIndex = R.FirstEventIndex;
            SyntheticRegionLastEventIndex = R.LastEventIndex;
            SyntheticObserved = R.bHasAudio ? FMath::Max(0.0f, R.AudioEndSec - R.AudioStartSec) : 0.0f;
            SyntheticPlanned = FMath::Max(0.001f, R.PlanEndSec - R.PlanStartSec);
            SyntheticScale = R.bHasAudio ? SyntheticObserved / SyntheticPlanned : GlobalScale;
            SyntheticCenter = R.bHasAudio
                ? (R.AudioStartSec + FMath::Max(0.0f, PlanCenter - R.PlanStartSec) * SyntheticScale)
                : GlobalCenter;
            ++SyntheticCoveredEventCount;
        }

        const float ActualMinusPlanMs = (ActualCenter - PlanCenter) * 1000.0f;
        const float ActualMinusGlobalMs = (ActualCenter - GlobalCenter) * 1000.0f;
        const float ActualMinusIslandMs = (ActualCenter - IslandCenter) * 1000.0f;
        const float ActualMinusSyntheticMs = (ActualCenter - SyntheticCenter) * 1000.0f;
        ActualAbsMs += std::abs(ActualMinusPlanMs);
        GlobalAbsMs += std::abs(ActualMinusGlobalMs);
        IslandAbsMs += std::abs(ActualMinusIslandMs);
        SyntheticAbsMs += std::abs(ActualMinusSyntheticMs);
        ++ComparedCount;

        const char* ModeHint = E.AudioIslandIndex == INDEX_NONE ? "fallback_or_missing_audio" : "speech_active_or_tail";
        Out << E.EventIndex << ','
            << CsvEscape(E.PoseID.ToString().ToStdString()) << ','
            << CsvEscape(E.SourceWord.ToStdString()) << ','
            << E.WordIndex << ','
            << E.TextIslandIndex << ','
            << E.AudioIslandIndex << ','
            << SyntheticRegionIndex << ','
            << SyntheticRegionFirstEventIndex << ','
            << SyntheticRegionLastEventIndex << ','
            << PlanCenter << ','
            << ActualCenter << ','
            << GlobalCenter << ','
            << IslandCenter << ','
            << SyntheticCenter << ','
            << ActualMinusPlanMs << ','
            << ActualMinusGlobalMs << ','
            << ActualMinusIslandMs << ','
            << ActualMinusSyntheticMs << ','
            << ObservedActiveSec << ','
            << PlannedActiveSec << ','
            << GlobalScale << ','
            << IslandObserved << ','
            << IslandPlanned << ','
            << IslandScale << ','
            << SyntheticObserved << ','
            << SyntheticPlanned << ','
            << SyntheticScale << ','
            << CsvEscape(E.CommitReason.ToString().ToStdString()) << ','
            << ModeHint << "\n";
    }

    std::ofstream Summary(SummaryPath);
    Summary << "PlannedEventCount,CommittedEventCount,ObservedSpeechIslandCount,SyntheticRegionCount,SyntheticRegionCoveredEventCount,PlannedActiveSec,ObservedActiveSec,GlobalScale,TailFlushCount,FallbackOrMissingAudioCount,ActualVsPlanMAEMs,ActualVsGlobalScaledMAEMs,ActualVsIslandScaledMAEMs,ActualVsSyntheticRegionScaledMAEMs\n";
    const double Den = ComparedCount > 0 ? static_cast<double>(ComparedCount) : 1.0;
    Summary << Plan.Events.Num() << ','
        << Track.Events.Num() << ','
        << AudioIslands.Num() << ','
        << SyntheticRegions.size() << ','
        << SyntheticCoveredEventCount << ','
        << PlannedActiveSec << ','
        << ObservedActiveSec << ','
        << GlobalScale << ','
        << TailFlushCount << ','
        << FallbackCount << ','
        << (ActualAbsMs / Den) << ','
        << (GlobalAbsMs / Den) << ','
        << (IslandAbsMs / Den) << ','
        << (SyntheticAbsMs / Den) << "\n";
}

static void WriteRuntimeCommitEventsCsv(const fs::path& Path, const FOffgridAIAlignedVisemeTrack& Track)
{
    std::ofstream Out(Path);
    Out << "LineID,EventIndex,TextIslandIndex,AudioIslandIndex,PoseID,SourceWord,FinalRenderCenterSec,PlaybackSecAtCommit,CommitLeadSec,CommitReason,ObservedAudioEndSec,SpeechIslandStartSec,SpeechIslandEndSec,CommittedTrackEventCount\n";
    for (const FOffgridAIAlignedVisemeEvent& E : Track.Events)
    {
        Out << CsvEscape(Track.LineID.ToString().ToStdString()) << ','
            << E.EventIndex << ','
            << E.TextIslandIndex << ','
            << E.AudioIslandIndex << ','
            << CsvEscape(E.PoseID.ToString().ToStdString()) << ','
            << CsvEscape(E.SourceWord.ToStdString()) << ','
            << E.FinalRenderCenterSeconds << ','
            << E.CommitPlaybackSeconds << ','
            << E.CommitLeadSeconds << ','
            << CsvEscape(E.CommitReason.ToString().ToStdString()) << ','
            << E.IslandAudioEndSeconds << ','
            << E.IslandAudioStartSeconds << ','
            << E.IslandAudioEndSeconds << ','
            << Track.Events.Num() << "\n";
    }
}

static void WriteAU39RuntimeTopologyMetricsCsv(const fs::path& Path, const FOffgridAIAlignedVisemeTrack& Track, float PrerollSec)
{
    std::vector<float> GapsMs;
    GapsMs.reserve(std::max(0, Track.Events.Num() - 1));
    int DenseGapCount = 0;
    int LargeGapCount = 0;
    int NegativeGapCount = 0;
    const float MinHealthyGapMs = std::max(25.0f, PrerollSec * 1000.0f * 0.18f);
    const float LargeGapMs = std::max(110.0f, PrerollSec * 1000.0f * 0.75f);
    for (int32 I = 1; I < Track.Events.Num(); ++I)
    {
        const float GapMs = (Track.Events[I].FinalRenderCenterSeconds - Track.Events[I - 1].FinalRenderCenterSeconds) * 1000.0f;
        GapsMs.push_back(GapMs);
        if (GapMs < -0.5f)
        {
            ++NegativeGapCount;
        }
        if (GapMs >= 0.0f && GapMs < MinHealthyGapMs)
        {
            ++DenseGapCount;
        }
        if (GapMs > LargeGapMs)
        {
            ++LargeGapCount;
        }
    }

    auto Percentile = [](std::vector<float> Values, float P) -> float
    {
        if (Values.empty())
        {
            return 0.0f;
        }
        std::sort(Values.begin(), Values.end());
        const float Pos = std::clamp(P, 0.0f, 1.0f) * static_cast<float>(Values.size() - 1);
        const size_t Lo = static_cast<size_t>(std::floor(Pos));
        const size_t Hi = std::min(Values.size() - 1, Lo + 1);
        const float T = Pos - static_cast<float>(Lo);
        return Values[Lo] * (1.0f - T) + Values[Hi] * T;
    };

    float SumAbsGapMs = 0.0f;
    float MaxGapMs = 0.0f;
    float MinGapMs = GapsMs.empty() ? 0.0f : std::numeric_limits<float>::max();
    for (float G : GapsMs)
    {
        SumAbsGapMs += std::abs(G);
        MaxGapMs = std::max(MaxGapMs, G);
        MinGapMs = std::min(MinGapMs, G);
    }

    const float MeanAbsGapMs = GapsMs.empty() ? 0.0f : SumAbsGapMs / static_cast<float>(GapsMs.size());
    std::ofstream Out(Path);
    Out << "AU39EventCount," << Track.Events.Num() << "\n";
    Out << "AU39GapCount," << GapsMs.size() << "\n";
    Out << "AU39MinEventGapMs," << MinGapMs << "\n";
    Out << "AU39P50EventGapMs," << Percentile(GapsMs, 0.50f) << "\n";
    Out << "AU39P90EventGapMs," << Percentile(GapsMs, 0.90f) << "\n";
    Out << "AU39MaxEventGapMs," << MaxGapMs << "\n";
    Out << "AU39MeanAbsEventGapMs," << MeanAbsGapMs << "\n";
    Out << "AU39DenseGapCount," << DenseGapCount << "\n";
    Out << "AU39LargeGapCount," << LargeGapCount << "\n";
    Out << "AU39NegativeGapCount," << NegativeGapCount << "\n";
    Out << "AU39MinHealthyGapMs," << MinHealthyGapMs << "\n";
    Out << "AU39LargeGapThresholdMs," << LargeGapMs << "\n";
    Out << "AU39RuntimeReflowDisabled," << 1 << "\n";
}

static void WriteAU41NoFragmentRuntimeMetricsCsv(
    const fs::path& Path,
    const FOffgridAIAlignedVisemeTrack& Track,
    const TArray<FOffgridAIStreamingSpeechIsland>& SpeechIslands,
    float PrerollSec)
{
    std::set<int32> TextIslands;
    std::set<int32> ReferencedAudioOnsets;
    int32 OnsetOnlyEvents = 0;
    int32 FragmentTimedEvents = 0;
    int32 MultiFragmentModeEvents = 0;
    for (const FOffgridAIAlignedVisemeEvent& E : Track.Events)
    {
        if (E.TextIslandIndex != INDEX_NONE)
        {
            TextIslands.insert(E.TextIslandIndex);
        }
        if (E.AudioIslandIndex != INDEX_NONE)
        {
            ReferencedAudioOnsets.insert(E.AudioIslandIndex);
        }
        const std::string Mode = E.AudioNudgeSearchMode.ToString().ToStdString();
        const std::string Reason = E.AudioNudgeRejectReason.ToString().ToStdString();
        if (Mode.find("AU41_onset_only") != std::string::npos)
        {
            ++OnsetOnlyEvents;
        }
        if (Mode.find("fragment") != std::string::npos || Reason.find("Fragment") != std::string::npos || Reason.find("ProsodyFragment") != std::string::npos)
        {
            ++FragmentTimedEvents;
        }
        if (Mode.find("prosody_fragment_assignment") != std::string::npos)
        {
            ++MultiFragmentModeEvents;
        }
    }

    std::ofstream Out(Path);
    Out << "Metric,Value\n";
    Out << "AU41DiagnosticsPresent,1\n";
    Out << "AU41RuntimeNoFragments,1\n";
    Out << "AU41EventCount," << Track.Events.Num() << "\n";
    Out << "AU41OnsetOnlyEventCount," << OnsetOnlyEvents << "\n";
    Out << "AU41FragmentTimedEventCount," << FragmentTimedEvents << "\n";
    Out << "AU41MultiFragmentModeEventCount," << MultiFragmentModeEvents << "\n";
    Out << "AU41TextIslandCount," << TextIslands.size() << "\n";
    Out << "AU41ReferencedOnsetCount," << ReferencedAudioOnsets.size() << "\n";
    Out << "AU41DetectorSpeechIslandCount," << SpeechIslands.Num() << "\n";
    Out << "AU41PrerollSec," << PrerollSec << "\n";
}




static void WriteLandmarkCandidatesCsv(const fs::path& Path, const TArray<FOffgridAIAudioLandmarkCandidate>& Rows)
{
    std::ofstream Out(Path);
    Out << "Class,TimeSec,Confidence,WindowStartSec,WindowEndSec,AvailableAtSec,Oracle\n";
    for (const FOffgridAIAudioLandmarkCandidate& Row : Rows)
    {
        Out << CsvEscape(FOffgridAIStreamingLandmarkDetector::ToString(Row.Class)) << ','
            << Row.TimeSec << ','
            << Row.Confidence << ','
            << Row.WindowStartSec << ','
            << Row.WindowEndSec << ','
            << Row.AvailableAtSec << ','
            << (Row.bOracle ? 1 : 0) << "\n";
    }
}



static void WriteLandmarkRawWindowsCsv(const fs::path& Path, const TArray<FOffgridAIAudioLandmarkRawWindow>& Rows)
{
    std::ofstream Out(Path);
    Out << "WindowStartSec,WindowEndSec,CenterSec,Rms,MeanAbs,Zcr,DiffMean,LowDominance,ProposedClass,ProposedConfidence,Accepted,RejectReason\n";
    for (const FOffgridAIAudioLandmarkRawWindow& Row : Rows)
    {
        Out << Row.WindowStartSec << ','
            << Row.WindowEndSec << ','
            << Row.CenterSec << ','
            << Row.Rms << ','
            << Row.MeanAbs << ','
            << Row.Zcr << ','
            << Row.DiffMean << ','
            << Row.LowDominance << ','
            << CsvEscape(FOffgridAIStreamingLandmarkDetector::ToString(Row.ProposedClass)) << ','
            << Row.ProposedConfidence << ','
            << (Row.bAccepted ? 1 : 0) << ','
            << CsvEscape(Row.RejectReason.ToString().ToStdString()) << "\n";
    }
}

static void WriteRuntimeLandmarkCorrespondenceCsv(const fs::path& Path, const FOffgridAIAlignedVisemeTrack& Track)
{
    std::ofstream Out(Path);
    Out << "LineID,EventIndex,PoseID,SourceWord,PlannedCenterSec,PlannedLandmarkClass,CandidateLandmarkClass,CompatibilityKind,CompatibilityScore,Matched,ActionableWithinPreroll,DetectedAudioTimeSec,DetectedTargetCenterSec,DeltaMs,Confidence,RequiredConfidence,AvailableAtSec,UsefulLeadMs,SearchStartSec,SearchEndSec,AudioWindowStartSec,AudioWindowEndSec,Reason\n";
    for (const FOffgridAIAlignedVisemeEvent& E : Track.Events)
    {
        if (!E.bAudioNudgeSearchPerformed && E.AudioNudgeSearchMode == NAME_None)
        {
            continue;
        }
        const EOffgridAIAudioLandmarkClass Class = FOffgridAIStreamingLandmarkDetector::LandmarkClassForPose(E.PoseID);
        if (Class == EOffgridAIAudioLandmarkClass::Unknown && !E.bAudioNudgeSearchPerformed)
        {
            continue;
        }
        const bool bMatched = E.bAudioNudgeCorrespondenceMatched
            || E.AudioNudgeRejectReason == FName(TEXT("G04_correspondence_match_observed_no_retime"))
            || E.AudioNudgeRejectReason == FName(TEXT("G02_correspondence_match_observed_no_retime"));
        const float DetectedAudioTimeSec = E.AudioNudgeCandidateAudioTimeSeconds > 0.0f
            ? E.AudioNudgeCandidateAudioTimeSeconds
            : (bMatched ? (E.AudioNudgeCandidateRawCenterSeconds + E.AudioNudgePredictedLeadSeconds) : 0.0f);
        Out << CsvEscape(Track.LineID.ToString().ToStdString()) << ','
            << E.EventIndex << ','
            << CsvEscape(E.PoseID.ToString().ToStdString()) << ','
            << CsvEscape(E.SourceWord.ToStdString()) << ','
            << E.AudioNudgeScheduledCenterSeconds << ','
            << CsvEscape(FOffgridAIStreamingLandmarkDetector::ToString(Class)) << ','
            << CsvEscape(E.AudioNudgeCandidateLandmarkClass.ToString().ToStdString()) << ','
            << CsvEscape(E.AudioNudgeCompatibilityKind.ToString().ToStdString()) << ','
            << E.AudioNudgeCompatibilityScore << ','
            << (bMatched ? 1 : 0) << ','
            << (E.bAudioNudgeActionableWithinPreroll ? 1 : 0) << ','
            << DetectedAudioTimeSec << ','
            << E.AudioNudgeCandidateRawCenterSeconds << ','
            << (E.AudioNudgeCandidateRawShiftSeconds * 1000.0f) << ','
            << E.AudioNudgeCandidateConfidence << ','
            << E.AudioNudgeRequiredConfidence << ','
            << E.AudioNudgeCandidateAvailableAtSeconds << ','
            << (E.AudioNudgeAvailableBeforeSearchSeconds * 1000.0f) << ','
            << E.AudioNudgeSearchStartSeconds << ','
            << E.AudioNudgeSearchEndSeconds << ','
            << E.AudioNudgeAvailableAudioStartSeconds << ','
            << E.AudioNudgeAvailableAudioEndSeconds << ','
            << CsvEscape(E.AudioNudgeRejectReason.ToString().ToStdString()) << "\n";
    }
}



struct FLandmarkDetectorScore
{
    int OracleCount = 0;
    int StreamingCandidateCount = 0;
    int MatchedCount = 0;
    int WrongClassCount = 0;
    int MissedStrongCount = 0;
    int AcousticLateCount = 0;
    int UsefulLateCount = 0;
    double TimingAbsErrorSumMs = 0.0;
    double TimingP95Ms = 0.0;
    double MeanAcousticLeadSec = 0.0;
    double MeanBufferedUsefulLeadSec = 0.0;
    int UsefulLeadGE25msCount = 0;
    int UsefulLeadGE50msCount = 0;
    int UsefulLeadGE75msCount = 0;
    int UsefulLeadGE100msCount = 0;
    std::vector<float> TimingErrorsMs;
};

static FLandmarkDetectorScore ScoreStreamingLandmarksAgainstOracle(
    const TArray<FOffgridAIAudioLandmarkCandidate>& Streaming,
    const TArray<FOffgridAIAudioLandmarkCandidate>& Oracle,
    float PrerollSec)
{
    FLandmarkDetectorScore Score;
    Score.OracleCount = Oracle.Num();
    Score.StreamingCandidateCount = Streaming.Num();
    const float MatchWindowSec = std::max(0.025f, PrerollSec * 0.75f);
    std::vector<bool> Used((size_t)Streaming.Num(), false);
    double AcousticLeadSum = 0.0;
    double BufferedUsefulLeadSum = 0.0;
    int LeadCount = 0;
    for (const FOffgridAIAudioLandmarkCandidate& O : Oracle)
    {
        int BestIndex = INDEX_NONE;
        float BestAbsDt = std::numeric_limits<float>::max();
        int BestWrongClassIndex = INDEX_NONE;
        float BestWrongClassAbsDt = std::numeric_limits<float>::max();
        for (int I = 0; I < Streaming.Num(); ++I)
        {
            if (Used[(size_t)I]) continue;
            const FOffgridAIAudioLandmarkCandidate& S = Streaming[I];
            const float AbsDt = std::fabs(S.TimeSec - O.TimeSec);
            if (AbsDt > MatchWindowSec) continue;
            if (S.Class == O.Class)
            {
                if (AbsDt < BestAbsDt) { BestAbsDt = AbsDt; BestIndex = I; }
            }
            else if (AbsDt < BestWrongClassAbsDt)
            {
                BestWrongClassAbsDt = AbsDt;
                BestWrongClassIndex = I;
            }
        }
        if (BestIndex != INDEX_NONE)
        {
            Used[(size_t)BestIndex] = true;
            ++Score.MatchedCount;
            const FOffgridAIAudioLandmarkCandidate& S = Streaming[BestIndex];
            const float ErrMs = (S.TimeSec - O.TimeSec) * 1000.0f;
            Score.TimingErrorsMs.push_back(std::fabs(ErrMs));
            Score.TimingAbsErrorSumMs += std::fabs(ErrMs);
            // AcousticLeadSec asks whether the local detector timestamp itself
            // precedes the oracle landmark. That is useful detector-debug info,
            // but it is NOT the runtime usefulness test when playback is delayed
            // by PrerollSec. BufferedUsefulLeadSec asks whether the candidate is
            // known before the landmark would render at playback time.
            const float AcousticLeadSec = O.TimeSec - S.AvailableAtSec;
            const float BufferedUsefulLeadSec = (O.TimeSec + PrerollSec) - S.AvailableAtSec;
            AcousticLeadSum += AcousticLeadSec;
            BufferedUsefulLeadSum += BufferedUsefulLeadSec;
            ++LeadCount;
            if (AcousticLeadSec < -std::max(0.005f, PrerollSec * 0.10f))
            {
                ++Score.AcousticLateCount;
            }
            if (BufferedUsefulLeadSec < 0.0f)
            {
                ++Score.UsefulLateCount;
            }
            if (BufferedUsefulLeadSec >= 0.025f) { ++Score.UsefulLeadGE25msCount; }
            if (BufferedUsefulLeadSec >= 0.050f) { ++Score.UsefulLeadGE50msCount; }
            if (BufferedUsefulLeadSec >= 0.075f) { ++Score.UsefulLeadGE75msCount; }
            if (BufferedUsefulLeadSec >= 0.100f) { ++Score.UsefulLeadGE100msCount; }
        }
        else
        {
            ++Score.MissedStrongCount;
            if (BestWrongClassIndex != INDEX_NONE && BestWrongClassAbsDt <= MatchWindowSec)
            {
                ++Score.WrongClassCount;
                Used[(size_t)BestWrongClassIndex] = true;
            }
        }
    }
    if (!Score.TimingErrorsMs.empty())
    {
        std::sort(Score.TimingErrorsMs.begin(), Score.TimingErrorsMs.end());
        const size_t P95Index = std::min(Score.TimingErrorsMs.size() - 1, (size_t)std::floor(0.95 * (Score.TimingErrorsMs.size() - 1)));
        Score.TimingP95Ms = Score.TimingErrorsMs[P95Index];
    }
    Score.MeanAcousticLeadSec = LeadCount > 0 ? AcousticLeadSum / LeadCount : 0.0;
    Score.MeanBufferedUsefulLeadSec = LeadCount > 0 ? BufferedUsefulLeadSum / LeadCount : 0.0;
    return Score;
}

static void WriteLandmarkDetectorScoreCsv(const fs::path& Path, const std::string& CaseId, const FLandmarkDetectorScore& Score)
{
    std::ofstream Out(Path);
    const double Precision = Score.StreamingCandidateCount > 0 ? static_cast<double>(Score.MatchedCount) / Score.StreamingCandidateCount : 0.0;
    const double Recall = Score.OracleCount > 0 ? static_cast<double>(Score.MatchedCount) / Score.OracleCount : 0.0;
    const double TimingMAE = Score.MatchedCount > 0 ? Score.TimingAbsErrorSumMs / Score.MatchedCount : 0.0;
    Out << "CaseId,OracleCount,StreamingCandidateCount,MatchedCount,Precision,Recall,TimingMAEMs,TimingP95Ms,AcousticLateCount,UsefulLateCount,WrongClassCount,MissedStrongCount,MeanAcousticLeadSec,MeanBufferedUsefulLeadSec,UsefulLeadGE25msCount,UsefulLeadGE50msCount,UsefulLeadGE75msCount,UsefulLeadGE100msCount\n";
    Out << CsvEscape(CaseId) << ',' << Score.OracleCount << ',' << Score.StreamingCandidateCount << ',' << Score.MatchedCount << ','
        << Precision << ',' << Recall << ',' << TimingMAE << ',' << Score.TimingP95Ms << ',' << Score.AcousticLateCount << ',' << Score.UsefulLateCount << ','
        << Score.WrongClassCount << ',' << Score.MissedStrongCount << ',' << Score.MeanAcousticLeadSec << ',' << Score.MeanBufferedUsefulLeadSec << ','
        << Score.UsefulLeadGE25msCount << ',' << Score.UsefulLeadGE50msCount << ',' << Score.UsefulLeadGE75msCount << ',' << Score.UsefulLeadGE100msCount << "\n";
}


static TArray<FOffgridAIAudioLandmarkCandidate> GenerateStreamingLandmarkCandidatesForPreroll(
    const FWavData& Wav,
    int ChunkMs,
    float PrerollSec)
{
    FOffgridAIStreamingLandmarkDetector Detector;
    Detector.Configure(PrerollSec);
    const int FramesPerChunk = std::max(1, (Wav.SampleRate * ChunkMs) / 1000);
    const int TotalFrames = static_cast<int>(Wav.SamplesInterleaved.size()) / std::max(1, Wav.Channels);
    for (int FrameOffset = 0; FrameOffset < TotalFrames; FrameOffset += FramesPerChunk)
    {
        const int FramesThisChunk = std::min(FramesPerChunk, TotalFrames - FrameOffset);
        TArray<uint8> ChunkBytes;
        ChunkBytes.Reserve(FramesThisChunk * Wav.Channels * 2);
        for (int F = 0; F < FramesThisChunk; ++F)
        {
            for (int C = 0; C < Wav.Channels; ++C)
            {
                const int16_t S = Wav.SamplesInterleaved[(FrameOffset + F) * Wav.Channels + C];
                ChunkBytes.Add(static_cast<uint8>(S & 0xff));
                ChunkBytes.Add(static_cast<uint8>((static_cast<uint16_t>(S) >> 8) & 0xff));
            }
        }
        Detector.PushPCM16(ChunkBytes, ChunkBytes.Num(), Wav.SampleRate, Wav.Channels, FrameOffset);
    }
    return Detector.GetCandidates();
}

static std::vector<float> BuildPrerollSweepSeconds(float RuntimePrerollSec)
{
    std::vector<float> Values = {0.050f, 0.075f, 0.100f, 0.125f, 0.150f, 0.175f, 0.200f, 0.250f};
    if (RuntimePrerollSec > 0.0f)
    {
        bool bExists = false;
        for (float V : Values)
        {
            if (std::fabs(V - RuntimePrerollSec) < 0.0005f)
            {
                bExists = true;
                break;
            }
        }
        if (!bExists)
        {
            Values.push_back(RuntimePrerollSec);
            std::sort(Values.begin(), Values.end());
        }
    }
    return Values;
}

static void WriteLandmarkPrerollSweepCsv(
    const fs::path& Path,
    const std::string& CaseId,
    const FWavData& Wav,
    int ChunkMs,
    float RuntimePrerollSec)
{
    std::ofstream Out(Path);
    Out << "CaseId,PrerollSec,PrerollMs,IsRuntimePreroll,OracleCount,StreamingCandidateCount,MatchedCount,Precision,Recall,TimingMAEMs,TimingP95Ms,AcousticLateCount,UsefulLateCount,MeanAcousticLeadSec,MeanBufferedUsefulLeadSec,UsefulLeadGE25msCount,UsefulLeadGE50msCount,UsefulLeadGE75msCount,UsefulLeadGE100msCount,UsefulLeadGE25msRate,UsefulLeadGE50msRate,UsefulLeadGE75msRate,UsefulLeadGE100msRate,AcousticLateRate,UsefulLateRate\n";
    const TArray<int16> Mono = BuildMonoPCM16(Wav);
    for (float PrerollSec : BuildPrerollSweepSeconds(RuntimePrerollSec))
    {
        const TArray<FOffgridAIAudioLandmarkCandidate> Streaming = GenerateStreamingLandmarkCandidatesForPreroll(Wav, ChunkMs, PrerollSec);
        const TArray<FOffgridAIAudioLandmarkCandidate> Oracle = FOffgridAIStreamingLandmarkDetector::GenerateOfflineOracle(Mono, Wav.SampleRate, PrerollSec);
        const FLandmarkDetectorScore Score = ScoreStreamingLandmarksAgainstOracle(Streaming, Oracle, PrerollSec);
        const double Precision = Score.StreamingCandidateCount > 0 ? static_cast<double>(Score.MatchedCount) / Score.StreamingCandidateCount : 0.0;
        const double Recall = Score.OracleCount > 0 ? static_cast<double>(Score.MatchedCount) / Score.OracleCount : 0.0;
        const double TimingMAE = Score.MatchedCount > 0 ? Score.TimingAbsErrorSumMs / Score.MatchedCount : 0.0;
        const double Denom = Score.MatchedCount > 0 ? static_cast<double>(Score.MatchedCount) : 0.0;
        Out << CsvEscape(CaseId) << ','
            << PrerollSec << ','
            << (PrerollSec * 1000.0f) << ','
            << (std::fabs(PrerollSec - RuntimePrerollSec) < 0.0005f ? 1 : 0) << ','
            << Score.OracleCount << ','
            << Score.StreamingCandidateCount << ','
            << Score.MatchedCount << ','
            << Precision << ','
            << Recall << ','
            << TimingMAE << ','
            << Score.TimingP95Ms << ','
            << Score.AcousticLateCount << ','
            << Score.UsefulLateCount << ','
            << Score.MeanAcousticLeadSec << ','
            << Score.MeanBufferedUsefulLeadSec << ','
            << Score.UsefulLeadGE25msCount << ','
            << Score.UsefulLeadGE50msCount << ','
            << Score.UsefulLeadGE75msCount << ','
            << Score.UsefulLeadGE100msCount << ','
            << (Denom > 0.0 ? Score.UsefulLeadGE25msCount / Denom : 0.0) << ','
            << (Denom > 0.0 ? Score.UsefulLeadGE50msCount / Denom : 0.0) << ','
            << (Denom > 0.0 ? Score.UsefulLeadGE75msCount / Denom : 0.0) << ','
            << (Denom > 0.0 ? Score.UsefulLeadGE100msCount / Denom : 0.0) << ','
            << (Denom > 0.0 ? Score.AcousticLateCount / Denom : 0.0) << ','
            << (Denom > 0.0 ? Score.UsefulLateCount / Denom : 0.0)
            << "\n";
    }
}

struct FAU37CommittedEventRef
{
    float CenterSec = 0.0f;
    int32 IslandIndex = INDEX_NONE;
    int32 GroupIndex = INDEX_NONE;
};

struct FAU37AnchorMatch
{
    int32 EventIndex = INDEX_NONE;
    int32 IslandIndex = INDEX_NONE;
    int32 GroupIndex = INDEX_NONE;
    FName PoseID = NAME_None;
    FString SourceWord;
    std::string LandmarkClass;
    float PlannedCenterSec = 0.0f;
    float CommittedCenterSec = 0.0f;
    bool bMatched = false;
    float CandidateTimeSec = 0.0f;
    float CandidateConfidence = 0.0f;
    float DeltaSec = 0.0f;
    float BufferedUsefulLeadSec = 0.0f;
};

static std::map<int32, FAU37CommittedEventRef> BuildAU37CommittedRefs(const FOffgridAIAlignedVisemeTrack& Track)
{
    std::map<int32, FAU37CommittedEventRef> Out;
    for (const FOffgridAIAlignedVisemeEvent& E : Track.Events)
    {
        FAU37CommittedEventRef Ref;
        Ref.CenterSec = E.FinalRenderCenterSeconds;
        Ref.IslandIndex = E.TextIslandIndex;
        Ref.GroupIndex = E.PlannerProsodyGroupIndex != INDEX_NONE ? E.PlannerProsodyGroupIndex : E.PhraseIndex;
        Out[E.EventIndex] = Ref;
    }
    return Out;
}

static std::vector<FAU37AnchorMatch> BuildAU37AnchorMatches(
    const FOffgridAITextVisemePlan& Plan,
    const FOffgridAIAlignedVisemeTrack& Track,
    const TArray<FOffgridAIAudioLandmarkCandidate>& Streaming,
    float PrerollSec)
{
    const float MatchWindowSec = std::max(0.025f, PrerollSec * 0.75f);
    const auto CommittedRefs = BuildAU37CommittedRefs(Track);
    std::vector<FAU37AnchorMatch> Rows;
    Rows.reserve(static_cast<size_t>(Plan.Events.Num()));
    for (int EventI = 0; EventI < Plan.Events.Num(); ++EventI)
    {
        const auto& E = Plan.Events[EventI];
        const EOffgridAIAudioLandmarkClass Class = FOffgridAIStreamingLandmarkDetector::LandmarkClassForPose(E.PoseID);
        if (Class == EOffgridAIAudioLandmarkClass::Unknown)
        {
            continue;
        }

        FAU37AnchorMatch Row;
        Row.EventIndex = EventI;
        Row.PoseID = E.PoseID;
        Row.SourceWord = E.SourceText;
        Row.IslandIndex = E.SentenceIslandIndex;
        Row.GroupIndex = E.PhraseIndex;
        Row.LandmarkClass = FOffgridAIStreamingLandmarkDetector::ToString(Class);
        Row.PlannedCenterSec = ((E.StartNorm + E.EndNorm) * 0.5f) * Plan.EstimatedDurationSeconds;
        Row.CommittedCenterSec = Row.PlannedCenterSec;
        const auto It = CommittedRefs.find(EventI);
        if (It != CommittedRefs.end())
        {
            Row.CommittedCenterSec = It->second.CenterSec;
            Row.IslandIndex = It->second.IslandIndex;
            Row.GroupIndex = It->second.GroupIndex;
        }

        int BestIndex = INDEX_NONE;
        float BestAbsDt = std::numeric_limits<float>::max();
        for (int I = 0; I < Streaming.Num(); ++I)
        {
            if (Streaming[I].Class != Class)
            {
                continue;
            }
            const float AbsDt = std::fabs(Streaming[I].TimeSec - Row.CommittedCenterSec);
            if (AbsDt < BestAbsDt && AbsDt <= MatchWindowSec)
            {
                BestAbsDt = AbsDt;
                BestIndex = I;
            }
        }
        if (BestIndex != INDEX_NONE)
        {
            const auto& C = Streaming[BestIndex];
            Row.bMatched = true;
            Row.CandidateTimeSec = C.TimeSec;
            Row.CandidateConfidence = C.Confidence;
            Row.DeltaSec = C.TimeSec - Row.CommittedCenterSec;
            Row.BufferedUsefulLeadSec = (Row.CommittedCenterSec + PrerollSec) - C.AvailableAtSec;
        }
        Rows.push_back(Row);
    }
    return Rows;
}

static void WritePlannedLandmarkCandidateMatchesCsv(
    const fs::path& Path,
    const FOffgridAITextVisemePlan& Plan,
    const FOffgridAIAlignedVisemeTrack& Track,
    const TArray<FOffgridAIAudioLandmarkCandidate>& Streaming,
    float PrerollSec)
{
    std::ofstream Out(Path);
    Out << "EventIndex,TextIslandIndex,ProsodyGroupIndex,PoseID,SourceWord,PlannedCenterSec,CommittedCenterSec,LandmarkClass,BestCandidateTimeSec,BestCandidateConfidence,DeltaMs,CommittedDeltaMs,AvailableAtSec,BufferedUsefulLeadSec,Matched,MatchedByCommittedCenter\n";
    const auto Rows = BuildAU37AnchorMatches(Plan, Track, Streaming, PrerollSec);
    for (const FAU37AnchorMatch& Row : Rows)
    {
        Out << Row.EventIndex << ','
            << Row.IslandIndex << ','
            << Row.GroupIndex << ','
            << CsvEscape(Row.PoseID.ToString().ToStdString()) << ','
            << CsvEscape(Row.SourceWord.ToStdString()) << ','
            << Row.PlannedCenterSec << ','
            << Row.CommittedCenterSec << ','
            << CsvEscape(Row.LandmarkClass) << ',';
        if (Row.bMatched)
        {
            Out << Row.CandidateTimeSec << ','
                << Row.CandidateConfidence << ','
                << ((Row.CandidateTimeSec - Row.PlannedCenterSec) * 1000.0f) << ','
                << (Row.DeltaSec * 1000.0f) << ','
                << (Row.CommittedCenterSec + PrerollSec - Row.BufferedUsefulLeadSec) << ','
                << Row.BufferedUsefulLeadSec << ",1,1\n";
        }
        else
        {
            Out << ",,,,,,,0,0\n";
        }
    }
}

static void WriteAU37ProsodyGroupAnchorReflowCsvs(
    const fs::path& AnchorPath,
    const fs::path& FeasibilityPath,
    const fs::path& ChainPath,
    const FOffgridAITextVisemePlan& Plan,
    const FOffgridAIAlignedVisemeTrack& Track,
    const TArray<FOffgridAIAudioLandmarkCandidate>& Streaming,
    float PrerollSec)
{
    const auto AnchorRows = BuildAU37AnchorMatches(Plan, Track, Streaming, PrerollSec);
    std::ofstream AnchorOut(AnchorPath);
    AnchorOut << "IslandIndex,GroupIndex,EventIndex,VisemeId,SourceWord,PlannedCenterSec,CommittedCenterSec,MatchedLandmarkClass,MatchedAudioSec,BufferedUsefulLeadSec,DeltaSec,MatchConfidence,UsableLead25,UsableLead50\n";
    for (const FAU37AnchorMatch& Row : AnchorRows)
    {
        AnchorOut << Row.IslandIndex << ',' << Row.GroupIndex << ',' << Row.EventIndex << ','
            << CsvEscape(Row.PoseID.ToString().ToStdString()) << ','
            << CsvEscape(Row.SourceWord.ToStdString()) << ','
            << Row.PlannedCenterSec << ',' << Row.CommittedCenterSec << ','
            << CsvEscape(Row.LandmarkClass) << ',';
        if (Row.bMatched)
        {
            AnchorOut << Row.CandidateTimeSec << ',' << Row.BufferedUsefulLeadSec << ',' << Row.DeltaSec << ',' << Row.CandidateConfidence << ','
                << (Row.BufferedUsefulLeadSec >= 0.025f ? 1 : 0) << ','
                << (Row.BufferedUsefulLeadSec >= 0.050f ? 1 : 0) << "\n";
        }
        else
        {
            AnchorOut << ",,,,0,0\n";
        }
    }

    struct FGroupInfo
    {
        int32 IslandIndex = INDEX_NONE;
        int32 GroupIndex = INDEX_NONE;
        int32 EventCount = 0;
        float StartSec = std::numeric_limits<float>::max();
        float EndSec = -std::numeric_limits<float>::max();
        std::vector<FAU37AnchorMatch> Anchors;
        float CorrectedStartSec = 0.0f;
        float CorrectedEndSec = 0.0f;
        float ImpliedShiftSec = 0.0f;
        float ImpliedScale = 1.0f;
        std::string ReflowMode = "none";
    };

    std::map<std::pair<int32, int32>, FGroupInfo> Groups;
    for (const FOffgridAIAlignedVisemeEvent& E : Track.Events)
    {
        const int32 GroupIndex = E.PlannerProsodyGroupIndex != INDEX_NONE ? E.PlannerProsodyGroupIndex : E.PhraseIndex;
        auto& G = Groups[{E.TextIslandIndex, GroupIndex}];
        G.IslandIndex = E.TextIslandIndex;
        G.GroupIndex = GroupIndex;
        ++G.EventCount;
        G.StartSec = std::min(G.StartSec, E.FinalRenderCenterSeconds);
        G.EndSec = std::max(G.EndSec, E.FinalRenderCenterSeconds);
    }
    for (const FAU37AnchorMatch& A : AnchorRows)
    {
        auto& G = Groups[{A.IslandIndex, A.GroupIndex}];
        G.IslandIndex = A.IslandIndex;
        G.GroupIndex = A.GroupIndex;
        G.Anchors.push_back(A);
    }

    std::ofstream FeasOut(FeasibilityPath);
    FeasOut << "IslandIndex,GroupIndex,GroupStartSec,GroupEndSec,GroupDurationSec,PlannedEventCount,StrongLandmarkCount,MatchedAnchorCount,UsableLead25AnchorCount,UsableLead50AnchorCount,AnchorCoverageRate,FirstAnchorDeltaSec,LastAnchorDeltaSec,AnchorSpanPlannedSec,AnchorSpanAudioSec,ImpliedScale,ImpliedShiftSec,CorrectedStartSec,CorrectedEndSec,ReflowMode\n";
    for (auto& Pair : Groups)
    {
        FGroupInfo& G = Pair.second;
        if (!std::isfinite(G.StartSec) || !std::isfinite(G.EndSec))
        {
            G.StartSec = 0.0f;
            G.EndSec = 0.0f;
        }
        std::vector<FAU37AnchorMatch> Matched;
        int Usable25 = 0;
        int Usable50 = 0;
        for (const FAU37AnchorMatch& A : G.Anchors)
        {
            if (A.bMatched)
            {
                Matched.push_back(A);
                if (A.BufferedUsefulLeadSec >= 0.025f) { ++Usable25; }
                if (A.BufferedUsefulLeadSec >= 0.050f) { ++Usable50; }
            }
        }
        std::sort(Matched.begin(), Matched.end(), [](const FAU37AnchorMatch& A, const FAU37AnchorMatch& B) { return A.CommittedCenterSec < B.CommittedCenterSec; });
        G.CorrectedStartSec = G.StartSec;
        G.CorrectedEndSec = G.EndSec;
        if (Matched.size() == 1)
        {
            G.ReflowMode = "single_anchor_shift";
            G.ImpliedShiftSec = Matched.front().DeltaSec;
            G.CorrectedStartSec = G.StartSec + G.ImpliedShiftSec;
            G.CorrectedEndSec = G.EndSec + G.ImpliedShiftSec;
        }
        else if (Matched.size() >= 2)
        {
            G.ReflowMode = Matched.size() == 2 ? "two_anchor_affine" : "multi_anchor_piecewise";
            const FAU37AnchorMatch& First = Matched.front();
            const FAU37AnchorMatch& Last = Matched.back();
            const float PlannedSpan = std::max(0.001f, Last.CommittedCenterSec - First.CommittedCenterSec);
            const float AudioSpan = std::max(0.001f, Last.CandidateTimeSec - First.CandidateTimeSec);
            G.ImpliedScale = AudioSpan / PlannedSpan;
            G.ImpliedShiftSec = First.CandidateTimeSec - (G.ImpliedScale * First.CommittedCenterSec);
            G.CorrectedStartSec = G.ImpliedShiftSec + G.ImpliedScale * G.StartSec;
            G.CorrectedEndSec = G.ImpliedShiftSec + G.ImpliedScale * G.EndSec;
        }
        const float FirstDelta = Matched.empty() ? 0.0f : Matched.front().DeltaSec;
        const float LastDelta = Matched.empty() ? 0.0f : Matched.back().DeltaSec;
        const float AnchorSpanPlanned = Matched.size() >= 2 ? (Matched.back().CommittedCenterSec - Matched.front().CommittedCenterSec) : 0.0f;
        const float AnchorSpanAudio = Matched.size() >= 2 ? (Matched.back().CandidateTimeSec - Matched.front().CandidateTimeSec) : 0.0f;
        FeasOut << G.IslandIndex << ',' << G.GroupIndex << ',' << G.StartSec << ',' << G.EndSec << ',' << std::max(0.0f, G.EndSec - G.StartSec) << ','
            << G.EventCount << ',' << G.Anchors.size() << ',' << Matched.size() << ',' << Usable25 << ',' << Usable50 << ','
            << (G.Anchors.empty() ? 0.0 : static_cast<double>(Matched.size()) / static_cast<double>(G.Anchors.size())) << ','
            << FirstDelta << ',' << LastDelta << ',' << AnchorSpanPlanned << ',' << AnchorSpanAudio << ','
            << G.ImpliedScale << ',' << G.ImpliedShiftSec << ',' << G.CorrectedStartSec << ',' << G.CorrectedEndSec << ',' << CsvEscape(G.ReflowMode) << "\n";
    }

    std::ofstream ChainOut(ChainPath);
    ChainOut << "IslandIndex,GroupIndex,NextGroupIndex,OriginalTailGapSec,CorrectedTailGapSec,RequiredNextShiftSec,WouldOverlap,WouldCreateDeadAir,ForwardPressureSec\n";
    const float MinGapSec = std::max(0.010f, PrerollSec / 15.0f);
    std::map<int32, std::vector<FGroupInfo>> ByIsland;
    for (const auto& Pair : Groups)
    {
        ByIsland[Pair.second.IslandIndex].push_back(Pair.second);
    }
    for (auto& Pair : ByIsland)
    {
        auto& Arr = Pair.second;
        std::sort(Arr.begin(), Arr.end(), [](const FGroupInfo& A, const FGroupInfo& B) { return A.StartSec < B.StartSec; });
        for (size_t I = 0; I + 1 < Arr.size(); ++I)
        {
            const FGroupInfo& A = Arr[I];
            const FGroupInfo& B = Arr[I + 1];
            const float OriginalGap = B.StartSec - A.EndSec;
            const float CorrectedGap = B.StartSec - A.CorrectedEndSec;
            const float RequiredShift = std::max(0.0f, (A.CorrectedEndSec + MinGapSec) - B.StartSec);
            const bool bOverlap = CorrectedGap < MinGapSec;
            const bool bDeadAir = CorrectedGap > std::max(0.080f, PrerollSec * 0.75f);
            const float ForwardPressure = A.CorrectedEndSec - A.EndSec;
            ChainOut << A.IslandIndex << ',' << A.GroupIndex << ',' << B.GroupIndex << ',' << OriginalGap << ',' << CorrectedGap << ','
                << RequiredShift << ',' << (bOverlap ? 1 : 0) << ',' << (bDeadAir ? 1 : 0) << ',' << ForwardPressure << "\n";
        }
    }
}

static void WriteSpeechIslandsCsv(const fs::path& Path, const TArray<FOffgridAIStreamingSpeechIsland>& Islands)
{
    std::ofstream Out(Path);
    Out << "IslandIndex,AudioBufferStartSec,AudioBufferLastSpeechSec,AudioBufferEndSec,Started,Ended\n";
    for (const FOffgridAIStreamingSpeechIsland& Island : Islands)
    {
        Out << Island.IslandIndex << ','
            << Island.AudioBufferStartSec << ','
            << Island.AudioBufferLastSpeechSec << ','
            << Island.AudioBufferEndSec << ','
            << (Island.bStarted ? 1 : 0) << ','
            << (Island.bEnded ? 1 : 0) << "\n";
    }
}


static void WriteAudioFeatureFramesCsv(const fs::path& Path, const TArray<FOffgridAIStreamingAudioFeatureFrame>& Frames)
{
    std::ofstream Out(Path);
    Out << "FrameIndex,AudioBufferStartSec,AudioBufferCenterSec,AudioBufferEndSec,RMS,RMSNorm,DeltaRMS,Flux,ZCR,LocalRMSPeak,LocalRMSValley,LocalFluxPeak\n";
    for (int32 I = 0; I < Frames.Num(); ++I)
    {
        const FOffgridAIStreamingAudioFeatureFrame& F = Frames[I];
        Out << I << ','
            << F.AudioBufferStartSec << ','
            << F.AudioBufferCenterSec << ','
            << F.AudioBufferEndSec << ','
            << F.RMS << ','
            << F.RMSNorm << ','
            << F.DeltaRMS << ','
            << F.Flux << ','
            << F.ZCR << ','
            << (F.bLocalRMSPeak ? 1 : 0) << ','
            << (F.bLocalRMSValley ? 1 : 0) << ','
            << (F.bLocalFluxPeak ? 1 : 0) << "\n";
    }
}

struct FFirstOnsetDiagnostic
{
    std::string Mode;
    bool bFound = false;
    bool bStrictVoicedGate = false;
    float SustainSeconds = 0.0f;
    float CandidateStartSec = 0.0f;
    float AcceptedStartSec = 0.0f;
    float AcceptFrameStartSec = 0.0f;
    float AcceptFrameEndSec = 0.0f;
    float CandidateAccumSec = 0.0f;
    float OpenThreshold = 0.0f;
    float NoiseFloorRMS = 0.0f;
    float RMS = 0.0f;
    float RMSNorm = 0.0f;
    float Flux = 0.0f;
    float ZCR = 0.0f;
    int32 AcceptFrameIndex = INDEX_NONE;
};

static bool Alpha13HasStrictVoicedEvidence(const FOffgridAIStreamingAudioFeatureFrame& F)
{
    return (F.ZCR <= 0.180f && F.Flux >= 0.010f)
        || (F.RMSNorm >= 0.42f && F.ZCR <= 0.240f);
}

static FFirstOnsetDiagnostic SimulateFirstOnsetFromFeatures(
    const TArray<FOffgridAIStreamingAudioFeatureFrame>& Frames,
    const std::string& Mode,
    bool bStrictVoicedGate,
    float OpenSustainSeconds,
    bool bBackdateToCandidate)
{
    FFirstOnsetDiagnostic Out;
    Out.Mode = Mode;
    Out.bStrictVoicedGate = bStrictVoicedGate;
    Out.SustainSeconds = OpenSustainSeconds;

    bool bCandidateActive = false;
    float CandidateStartSec = 0.0f;
    float CandidateAccumSec = 0.0f;
    float NoiseFloorRMS = 0.0001f;
    float SpeechPeakRMS = 0.0001f;

    for (int32 I = 0; I < Frames.Num(); ++I)
    {
        const FOffgridAIStreamingAudioFeatureFrame& F = Frames[I];
        SpeechPeakRMS = FMath::Max(0.0001f, SpeechPeakRMS * 0.970f);
        NoiseFloorRMS = FMath::Clamp(NoiseFloorRMS * 0.980f + F.RMS * 0.020f, 0.000001f, 0.050f);
        const float OpenThreshold = FMath::Max(0.004f, NoiseFloorRMS * 3.0f + 0.003f);
        const bool bVoicedOk = !bStrictVoicedGate || Alpha13HasStrictVoicedEvidence(F);
        const bool bOpenFrame = F.RMS >= OpenThreshold && bVoicedOk;
        const float FrameDuration = FMath::Max(F.AudioBufferEndSec - F.AudioBufferStartSec, 0.0f);

        if (bOpenFrame)
        {
            if (!bCandidateActive)
            {
                bCandidateActive = true;
                CandidateStartSec = F.AudioBufferStartSec;
                CandidateAccumSec = 0.0f;
            }
            CandidateAccumSec += FrameDuration;
            if (CandidateAccumSec >= OpenSustainSeconds)
            {
                Out.bFound = true;
                Out.CandidateStartSec = CandidateStartSec;
                Out.AcceptedStartSec = bBackdateToCandidate ? CandidateStartSec : FMath::Max(0.0f, F.AudioBufferEndSec - OpenSustainSeconds);
                Out.AcceptFrameStartSec = F.AudioBufferStartSec;
                Out.AcceptFrameEndSec = F.AudioBufferEndSec;
                Out.CandidateAccumSec = CandidateAccumSec;
                Out.OpenThreshold = OpenThreshold;
                Out.NoiseFloorRMS = NoiseFloorRMS;
                Out.RMS = F.RMS;
                Out.RMSNorm = F.RMSNorm;
                Out.Flux = F.Flux;
                Out.ZCR = F.ZCR;
                Out.AcceptFrameIndex = I;
                return Out;
            }
        }
        else
        {
            bCandidateActive = false;
            CandidateAccumSec = 0.0f;
        }
    }

    Out.NoiseFloorRMS = NoiseFloorRMS;
    return Out;
}

static void WriteFirstOnsetComparisonCsv(
    const fs::path& Path,
    const TArray<FOffgridAIStreamingAudioFeatureFrame>& Frames,
    const TArray<FOffgridAIStreamingSpeechIsland>& ActualIslands)
{
    const FFirstOnsetDiagnostic Legacy = SimulateFirstOnsetFromFeatures(Frames, "legacy_20ms_energy_backdated", false, 0.020f, true);
    const FFirstOnsetDiagnostic Strict = SimulateFirstOnsetFromFeatures(Frames, "strict_55ms_voiced_no_backdate", true, 0.055f, false);

    float ActualFirstStartSec = 0.0f;
    bool bHasActualFirstStart = false;
    if (ActualIslands.Num() > 0 && ActualIslands[0].bStarted)
    {
        bHasActualFirstStart = true;
        ActualFirstStartSec = ActualIslands[0].AudioBufferStartSec;
    }

    const float StrictMinusLegacyAcceptedMs = (Strict.bFound && Legacy.bFound) ? (Strict.AcceptedStartSec - Legacy.AcceptedStartSec) * 1000.0f : 0.0f;
    const float StrictMinusLegacyCandidateMs = (Strict.bFound && Legacy.bFound) ? (Strict.CandidateStartSec - Legacy.CandidateStartSec) * 1000.0f : 0.0f;

    std::ofstream Out(Path);
    Out << "Mode,Found,StrictVoicedGate,SustainSec,BackdateToCandidate,CandidateStartSec,AcceptedStartSec,AcceptFrameStartSec,AcceptFrameEndSec,CandidateAccumSec,OpenThreshold,NoiseFloorRMS,RMS,RMSNorm,Flux,ZCR,AcceptFrameIndex,ActualFirstSpeechStartSec,ActualMinusModeAcceptedMs,StrictMinusLegacyAcceptedMs,StrictMinusLegacyCandidateMs\n";
    const FFirstOnsetDiagnostic Rows[2] = {Legacy, Strict};
    for (int32 I = 0; I < 2; ++I)
    {
        const FFirstOnsetDiagnostic& R = Rows[I];
        const bool bBackdate = R.Mode == "legacy_20ms_energy_backdated";
        const float ActualMinusModeMs = (bHasActualFirstStart && R.bFound) ? (ActualFirstStartSec - R.AcceptedStartSec) * 1000.0f : 0.0f;
        Out << CsvEscape(R.Mode) << ','
            << (R.bFound ? 1 : 0) << ','
            << (R.bStrictVoicedGate ? 1 : 0) << ','
            << R.SustainSeconds << ','
            << (bBackdate ? 1 : 0) << ','
            << R.CandidateStartSec << ','
            << R.AcceptedStartSec << ','
            << R.AcceptFrameStartSec << ','
            << R.AcceptFrameEndSec << ','
            << R.CandidateAccumSec << ','
            << R.OpenThreshold << ','
            << R.NoiseFloorRMS << ','
            << R.RMS << ','
            << R.RMSNorm << ','
            << R.Flux << ','
            << R.ZCR << ','
            << R.AcceptFrameIndex << ','
            << (bHasActualFirstStart ? ActualFirstStartSec : 0.0f) << ','
            << ActualMinusModeMs << ','
            << StrictMinusLegacyAcceptedMs << ','
            << StrictMinusLegacyCandidateMs << "\n";
    }
}
static bool IsDialogueWordCharForDiagnostics(TCHAR C)
{
    return FChar::IsAlnum(C) || C == TEXT('\'') || C == TEXT('-');
}

static bool IsHardSentencePunctuationForDiagnostics(TCHAR C)
{
    return C == TEXT('.') || C == TEXT('!') || C == TEXT('?') || C == TEXT('\n') || C == TEXT('\r');
}

static std::string PunctuationDebugName(TCHAR C)
{
    switch (C)
    {
    case TEXT('.'):
        return "period";
    case TEXT('!'):
        return "exclamation";
    case TEXT('?'):
        return "question";
    case TEXT(','):
        return "comma";
    case TEXT(';'):
        return "semicolon";
    case TEXT(':'):
        return "colon";
    case TEXT('\n'):
        return "newline";
    case TEXT('\r'):
        return "carriage_return";
    default:
        break;
    }
    std::string S;
    S.push_back(static_cast<char>(C));
    return S;
}

struct FBoundaryPunctuationDiagnostic
{
    bool bFound = false;
    int32 CharIndex = INDEX_NONE;
    TCHAR Character = 0;
    bool bHardSentenceBreak = false;
    std::string Kind = "none";
};

static std::map<int32, FBoundaryPunctuationDiagnostic> BuildWordBoundaryPunctuationDiagnostics(const std::string& Transcript)
{
    std::map<int32, FBoundaryPunctuationDiagnostic> ByWordIndex;
    const FString Text(Transcript.c_str());
    int32 WordIndex = 0;
    bool bInsideWord = false;
    FBoundaryPunctuationDiagnostic Pending;

    for (int32 I = 0; I < Text.Len(); ++I)
    {
        const TCHAR C = Text[I];
        if (IsDialogueWordCharForDiagnostics(C))
        {
            if (!bInsideWord)
            {
                if (Pending.bFound && WordIndex > 0)
                {
                    ByWordIndex[WordIndex] = Pending;
                }
                Pending = FBoundaryPunctuationDiagnostic();
                bInsideWord = true;
            }
            continue;
        }

        if (bInsideWord)
        {
            ++WordIndex;
            bInsideWord = false;
        }

        if (C == TEXT('.') || C == TEXT('!') || C == TEXT('?') || C == TEXT(',') || C == TEXT(';') || C == TEXT(':') || C == TEXT('\n') || C == TEXT('\r'))
        {
            Pending.bFound = true;
            Pending.CharIndex = I;
            Pending.Character = C;
            Pending.bHardSentenceBreak = IsHardSentencePunctuationForDiagnostics(C);
            Pending.Kind = PunctuationDebugName(C);
        }
    }

    return ByWordIndex;
}


static std::string ToLowerAscii(std::string S)
{
    for (char& C : S)
    {
        C = static_cast<char>(std::tolower(static_cast<unsigned char>(C)));
    }
    return S;
}

static bool IsAnyOf(const std::string& Word, const std::vector<std::string>& Values)
{
    const std::string Lower = ToLowerAscii(Word);
    return std::find(Values.begin(), Values.end(), Lower) != Values.end();
}

static std::string SoftBoundaryStructureClass(
    const std::string& PunctuationKind,
    const std::string& CurrFirstWord,
    int32 PrevWordCount,
    int32 CurrWordCount,
    int32 BoundaryOrdinalInIsland,
    int32 IslandSoftBoundaryCount)
{
    if (IsAnyOf(CurrFirstWord, {"and", "or", "nor"}))
    {
        if (PunctuationKind == "comma" && PrevWordCount <= 3 && CurrWordCount <= 3 && IslandSoftBoundaryCount >= 2)
        {
            return "compact_list";
        }
        return "coordinating_conjunction";
    }
    if (IsAnyOf(CurrFirstWord, {"but", "so", "yet"}))
    {
        return "contrast_or_result_conjunction";
    }
    if (IsAnyOf(CurrFirstWord, {"because", "although", "though", "while", "when", "where", "if", "unless", "until", "since", "after", "before"}))
    {
        return "subordinate_clause";
    }
    if (IsAnyOf(CurrFirstWord, {"who", "whom", "whose", "which", "that"}))
    {
        return "relative_clause";
    }
    if (PunctuationKind == "comma" && PrevWordCount <= 3 && CurrWordCount <= 3 && IslandSoftBoundaryCount >= 2)
    {
        return "compact_list";
    }
    if (BoundaryOrdinalInIsland == 1)
    {
        return "first_soft_boundary";
    }
    if (BoundaryOrdinalInIsland == IslandSoftBoundaryCount && IslandSoftBoundaryCount > 1)
    {
        return "last_soft_boundary";
    }
    return "other_soft_boundary";
}

static std::string JoinReasons(const std::vector<std::string>& Reasons)
{
    std::string Out;
    for (size_t I = 0; I < Reasons.size(); ++I)
    {
        if (I > 0)
        {
            Out += ";";
        }
        Out += Reasons[I];
    }
    return Out;
}

static void WriteTextIslandDiagnosticsCsv(
    const fs::path& Path,
    const std::string& Transcript,
    const FOffgridAITextVisemePlan& Plan,
    const FOffgridAIAlignedVisemeTrack& Track)
{
    struct FRuntimeIslandSummary
    {
        int32 TextIslandIndex = INDEX_NONE;
        int32 FirstEventIndex = INDEX_NONE;
        int32 LastEventIndex = INDEX_NONE;
        int32 EventCount = 0;
        int32 FirstWordIndex = INDEX_NONE;
        int32 LastWordIndex = INDEX_NONE;
        int32 FirstPhraseIndex = INDEX_NONE;
        int32 LastPhraseIndex = INDEX_NONE;
        int32 FirstPlannerSentenceIslandIndex = INDEX_NONE;
        int32 LastPlannerSentenceIslandIndex = INDEX_NONE;
        float FirstTextCenterSec = 0.0f;
        float LastTextCenterSec = 0.0f;
        float FirstCommittedCenterSec = 0.0f;
        float LastCommittedCenterSec = 0.0f;
        FString FirstWord;
        FString LastWord;
    };

    std::map<int32, FRuntimeIslandSummary> ByTextIsland;
    for (const FOffgridAIAlignedVisemeEvent& Event : Track.Events)
    {
        FRuntimeIslandSummary& Summary = ByTextIsland[Event.TextIslandIndex];
        const bool bFirst = Summary.EventCount == 0;
        Summary.TextIslandIndex = Event.TextIslandIndex;
        if (bFirst)
        {
            Summary.FirstEventIndex = Event.EventIndex;
            Summary.FirstWordIndex = Event.WordIndex;
            Summary.FirstPhraseIndex = Event.PhraseIndex;
            Summary.FirstTextCenterSec = Event.TextDiagnosticCenterSeconds;
            Summary.FirstCommittedCenterSec = Event.FinalRenderCenterSeconds;
            Summary.FirstWord = Event.SourceWord;
            if (Plan.Events.IsValidIndex(Event.EventIndex))
            {
                Summary.FirstPlannerSentenceIslandIndex = Plan.Events[Event.EventIndex].SentenceIslandIndex;
            }
        }
        Summary.LastEventIndex = Event.EventIndex;
        Summary.LastWordIndex = Event.WordIndex;
        Summary.LastPhraseIndex = Event.PhraseIndex;
        Summary.LastTextCenterSec = Event.TextDiagnosticCenterSeconds;
        Summary.LastCommittedCenterSec = Event.FinalRenderCenterSeconds;
        Summary.LastWord = Event.SourceWord;
        if (Plan.Events.IsValidIndex(Event.EventIndex))
        {
            Summary.LastPlannerSentenceIslandIndex = Plan.Events[Event.EventIndex].SentenceIslandIndex;
        }
        ++Summary.EventCount;
    }

    const std::map<int32, FBoundaryPunctuationDiagnostic> BoundaryByWordIndex = BuildWordBoundaryPunctuationDiagnostics(Transcript);

    std::ofstream Out(Path);
    Out << "RuntimeTextIslandIndex,FirstEventIndex,LastEventIndex,EventCount,FirstWord,LastWord,FirstWordIndex,LastWordIndex,FirstPhraseIndex,LastPhraseIndex,FirstPlannerSentenceIslandIndex,LastPlannerSentenceIslandIndex,BoundaryPrevEventIndex,BoundaryPrevWord,BoundaryPrevWordIndex,BoundaryPrevPhraseIndex,BoundaryPrevPlannerSentenceIslandIndex,BoundaryPhraseChanged,BoundaryPlannerSentenceChanged,BoundaryPunctuationKind,BoundaryPunctuationCharIndex,BoundaryIsHardSentenceBreak,PrimaryBoundaryReason,AllBoundaryReasons,FirstTextCenterSec,LastTextCenterSec,FirstCommittedCenterSec,LastCommittedCenterSec\n";

    for (const auto& Pair : ByTextIsland)
    {
        const FRuntimeIslandSummary& S = Pair.second;
        int32 PrevEventIndex = S.FirstEventIndex - 1;
        FString PrevWord;
        int32 PrevWordIndex = INDEX_NONE;
        int32 PrevPhraseIndex = INDEX_NONE;
        int32 PrevSentenceIslandIndex = INDEX_NONE;
        bool bPhraseChanged = false;
        bool bSentenceChanged = false;
        FBoundaryPunctuationDiagnostic Punctuation;
        std::vector<std::string> Reasons;

        if (S.FirstEventIndex <= 0)
        {
            PrevEventIndex = INDEX_NONE;
            Reasons.push_back("start_of_line");
        }
        else if (Plan.Events.IsValidIndex(PrevEventIndex) && Plan.Events.IsValidIndex(S.FirstEventIndex))
        {
            const FOffgridAITextVisemeEvent& Prev = Plan.Events[PrevEventIndex];
            const FOffgridAITextVisemeEvent& Curr = Plan.Events[S.FirstEventIndex];
            PrevWord = Prev.SourceText;
            PrevWordIndex = Prev.WordIndex;
            PrevPhraseIndex = Prev.PhraseIndex;
            PrevSentenceIslandIndex = Prev.SentenceIslandIndex;
            bPhraseChanged = Prev.PhraseIndex != Curr.PhraseIndex;
            bSentenceChanged = Prev.SentenceIslandIndex != Curr.SentenceIslandIndex;

            auto PuncIt = BoundaryByWordIndex.find(Curr.WordIndex);
            if (PuncIt != BoundaryByWordIndex.end())
            {
                Punctuation = PuncIt->second;
            }

            if (bSentenceChanged)
            {
                Reasons.push_back("planner_sentence_island_change");
            }
            if (Punctuation.bFound && Punctuation.bHardSentenceBreak)
            {
                Reasons.push_back("hard_sentence_punctuation_before_word");
            }
            if (bPhraseChanged)
            {
                Reasons.push_back("phrase_index_change");
            }
            if (Punctuation.bFound && !Punctuation.bHardSentenceBreak)
            {
                Reasons.push_back("soft_punctuation_before_word");
            }
            if (Reasons.empty())
            {
                Reasons.push_back("runtime_split_without_visible_planner_boundary");
            }
        }
        else
        {
            Reasons.push_back("invalid_event_index_boundary");
        }

        const std::string PrimaryReason = Reasons.empty() ? "unknown" : Reasons.front();

        Out << S.TextIslandIndex << ','
            << S.FirstEventIndex << ','
            << S.LastEventIndex << ','
            << S.EventCount << ','
            << CsvEscape(S.FirstWord.ToStdString()) << ','
            << CsvEscape(S.LastWord.ToStdString()) << ','
            << S.FirstWordIndex << ','
            << S.LastWordIndex << ','
            << S.FirstPhraseIndex << ','
            << S.LastPhraseIndex << ','
            << S.FirstPlannerSentenceIslandIndex << ','
            << S.LastPlannerSentenceIslandIndex << ','
            << PrevEventIndex << ','
            << CsvEscape(PrevWord.ToStdString()) << ','
            << PrevWordIndex << ','
            << PrevPhraseIndex << ','
            << PrevSentenceIslandIndex << ','
            << (bPhraseChanged ? 1 : 0) << ','
            << (bSentenceChanged ? 1 : 0) << ','
            << CsvEscape(Punctuation.Kind) << ','
            << Punctuation.CharIndex << ','
            << (Punctuation.bHardSentenceBreak ? 1 : 0) << ','
            << CsvEscape(PrimaryReason) << ','
            << CsvEscape(JoinReasons(Reasons)) << ','
            << S.FirstTextCenterSec << ','
            << S.LastTextCenterSec << ','
            << S.FirstCommittedCenterSec << ','
            << S.LastCommittedCenterSec << "\n";
    }
}


struct FPeakInfo
{
    float PeakTime = 0.0f;
    float PeakWeight = -1.0f;
    float Center = 0.0f;
    float PeakWindowStart = 0.0f;
    float PeakWindowEnd = 0.0f;
    float PeakWeightedTimeSum = 0.0f;
    float PeakWeightSum = 0.0f;
    int PeakSampleCount = 0;
};

struct FSubmittedPoseDiagnostics
{
    std::map<int, FPeakInfo> Peaks;
    float MaxVisiblePoseGapSec = 0.0f;
    int VisiblePoseGapCountOver250ms = 0;
};

static FSubmittedPoseDiagnostics CollectSubmittedPosePeaks(
    const FOffgridAIAlignedVisemeTrack& Track,
    float DurationSec,
    int Fps,
    float ActiveSpeechStartSec,
    float ActiveSpeechEndSec,
    std::ostream* Out)
{
    FSubmittedPoseDiagnostics Diagnostics;
    if (Out)
    {
        *Out << "PlaybackSec,EventIndex,PoseID,SourceWord,CommittedCenterSec,SubmittedWeight,SourceStrength\n";
    }
    const float Dt = 1.0f / static_cast<float>(std::max(1, Fps));
    const float EndSec = std::max(DurationSec + 0.75f, Track.Events.Num() > 0 ? Track.Events.Last().FinalRenderCenterSeconds + 0.75f : DurationSec);
    constexpr float VisiblePoseThreshold = 0.10f;
    constexpr float LongVisibleGapThresholdSec = 0.250f;
    const float ActiveStartSec = FMath::Clamp(ActiveSpeechStartSec, 0.0f, DurationSec);
    const float ActiveEndSec = FMath::Clamp(FMath::Max(ActiveSpeechEndSec, ActiveStartSec), ActiveStartSec, DurationSec);
    bool bInVisibleGap = false;
    float CurrentGapStartSec = ActiveStartSec;

    auto FinalizeGap = [&](float GapStartSec, float GapEndSec)
    {
        const float GapSec = FMath::Max(GapEndSec - GapStartSec, 0.0f);
        Diagnostics.MaxVisiblePoseGapSec = std::max(Diagnostics.MaxVisiblePoseGapSec, GapSec);
        if (GapSec > LongVisibleGapThresholdSec)
        {
            ++Diagnostics.VisiblePoseGapCountOver250ms;
        }
    };

    constexpr float PeakWindowEpsilon = 0.0025f;
    for (float T = 0.0f; T <= EndSec; T += Dt)
    {
        const TArray<FOffgridAISubmittedVisemeSample> Samples = FOffgridAIVisemePerformer::Sample(Track, T, false);
        bool bHasVisiblePose = false;
        for (const auto& S : Samples)
        {
            if (Out)
            {
                *Out << T << ','
                    << S.EventIndex << ','
                    << CsvEscape(S.PoseID.ToString().ToStdString()) << ','
                    << CsvEscape(S.SourceWord.ToStdString()) << ','
                    << S.CommittedRenderCenterSeconds << ','
                    << S.SubmittedWeight << ','
                    << S.SourceStrength << "\n";
            }
            if (S.SubmittedWeight > VisiblePoseThreshold)
            {
                bHasVisiblePose = true;
            }
            FPeakInfo& P = Diagnostics.Peaks[S.EventIndex];
            P.Center = S.CommittedRenderCenterSeconds;
            if (S.SubmittedWeight > P.PeakWeight + PeakWindowEpsilon)
            {
                P.PeakWeight = S.SubmittedWeight;
                P.PeakTime = T;
                P.PeakWindowStart = T;
                P.PeakWindowEnd = T;
                P.PeakWeightedTimeSum = T * S.SubmittedWeight;
                P.PeakWeightSum = S.SubmittedWeight;
                P.PeakSampleCount = 1;
            }
            else if (P.PeakWeight >= 0.0f && std::abs(S.SubmittedWeight - P.PeakWeight) <= PeakWindowEpsilon)
            {
                if (P.PeakSampleCount <= 0)
                {
                    P.PeakWindowStart = T;
                    P.PeakWindowEnd = T;
                    P.PeakWeightedTimeSum = T * S.SubmittedWeight;
                    P.PeakWeightSum = S.SubmittedWeight;
                    P.PeakSampleCount = 1;
                }
                else
                {
                    P.PeakWindowEnd = T;
                    P.PeakWeightedTimeSum += T * S.SubmittedWeight;
                    P.PeakWeightSum += S.SubmittedWeight;
                    ++P.PeakSampleCount;
                }
            }
        }

        if (T < ActiveStartSec || T > ActiveEndSec)
        {
            continue;
        }

        if (bHasVisiblePose)
        {
            if (bInVisibleGap)
            {
                FinalizeGap(CurrentGapStartSec, T);
                bInVisibleGap = false;
            }
        }
        else if (!bInVisibleGap)
        {
            bInVisibleGap = true;
            CurrentGapStartSec = T;
        }
    }

    if (ActiveEndSec > ActiveStartSec && bInVisibleGap)
    {
        FinalizeGap(CurrentGapStartSec, ActiveEndSec);
    }

    return Diagnostics;
}

static FSubmittedPoseDiagnostics WriteSubmittedPosesCsv(
    const fs::path& Path,
    const FOffgridAIAlignedVisemeTrack& Track,
    float DurationSec,
    int Fps,
    float ActiveSpeechStartSec,
    float ActiveSpeechEndSec)
{
    std::ofstream Out(Path);
    return CollectSubmittedPosePeaks(Track, DurationSec, Fps, ActiveSpeechStartSec, ActiveSpeechEndSec, &Out);
}

static float PeakWindowMidTime(const FPeakInfo& Peak)
{
    if (Peak.PeakSampleCount <= 0)
    {
        return Peak.PeakTime;
    }
    return 0.5f * (Peak.PeakWindowStart + Peak.PeakWindowEnd);
}

static float PeakWindowWeightedCenterTime(const FPeakInfo& Peak)
{
    if (Peak.PeakWeightSum <= 0.0f)
    {
        return PeakWindowMidTime(Peak);
    }
    return Peak.PeakWeightedTimeSum / Peak.PeakWeightSum;
}

static float PeakWindowWidthSec(const FPeakInfo& Peak)
{
    if (Peak.PeakSampleCount <= 1)
    {
        return 0.0f;
    }
    return FMath::Max(Peak.PeakWindowEnd - Peak.PeakWindowStart, 0.0f);
}

static float GetSpeechIslandEndSec(const FOffgridAIStreamingSpeechIsland& Island, float ObservedAudioEndSec)
{
    if (Island.bEnded && Island.AudioBufferEndSec > Island.AudioBufferStartSec)
    {
        return Island.AudioBufferEndSec;
    }
    if (!Island.bEnded)
    {
        return FMath::Max(Island.AudioBufferStartSec, ObservedAudioEndSec);
    }
    if (Island.AudioBufferLastSpeechSec > Island.AudioBufferStartSec)
    {
        return Island.AudioBufferLastSpeechSec;
    }
    return FMath::Max(Island.AudioBufferStartSec, ObservedAudioEndSec);
}

static bool IsCenterInsideSpeechIsland(
    const TArray<FOffgridAIStreamingSpeechIsland>& Islands,
    float CenterSec,
    float ObservedAudioEndSec,
    float EarlySlackSec,
    float LateSlackSec)
{
    for (const FOffgridAIStreamingSpeechIsland& Island : Islands)
    {
        if (!Island.bStarted)
        {
            continue;
        }

        const float IslandEndSec = GetSpeechIslandEndSec(Island, ObservedAudioEndSec);
        if (CenterSec >= Island.AudioBufferStartSec - EarlySlackSec && CenterSec <= IslandEndSec + LateSlackSec)
        {
            return true;
        }
    }
    return false;
}

static float DistanceToNearestSpeechIslandMs(
    const TArray<FOffgridAIStreamingSpeechIsland>& Islands,
    float CenterSec,
    float ObservedAudioEndSec)
{
    float BestDistanceSec = std::numeric_limits<float>::max();
    for (const FOffgridAIStreamingSpeechIsland& Island : Islands)
    {
        if (!Island.bStarted)
        {
            continue;
        }

        const float IslandEndSec = GetSpeechIslandEndSec(Island, ObservedAudioEndSec);
        float DistanceSec = 0.0f;
        if (CenterSec < Island.AudioBufferStartSec)
        {
            DistanceSec = Island.AudioBufferStartSec - CenterSec;
        }
        else if (CenterSec > IslandEndSec)
        {
            DistanceSec = CenterSec - IslandEndSec;
        }
        BestDistanceSec = FMath::Min(BestDistanceSec, DistanceSec);
    }
    return BestDistanceSec == std::numeric_limits<float>::max() ? 0.0f : BestDistanceSec * 1000.0f;
}

static std::optional<float> FindNextSpeechIslandStartSec(
    const TArray<FOffgridAIStreamingSpeechIsland>& Islands,
    float AfterSec)
{
    std::optional<float> Best;
    for (const FOffgridAIStreamingSpeechIsland& Island : Islands)
    {
        if (!Island.bStarted || Island.AudioBufferStartSec < AfterSec)
        {
            continue;
        }
        if (!Best || Island.AudioBufferStartSec < *Best)
        {
            Best = Island.AudioBufferStartSec;
        }
    }
    return Best;
}

static float LateDistancePastNearestSpeechIslandEndMs(
    const TArray<FOffgridAIStreamingSpeechIsland>& Islands,
    float CenterSec,
    float ObservedAudioEndSec)
{
    float BestLateDistanceSec = std::numeric_limits<float>::max();
    for (const FOffgridAIStreamingSpeechIsland& Island : Islands)
    {
        if (!Island.bStarted)
        {
            continue;
        }

        const float IslandEndSec = GetSpeechIslandEndSec(Island, ObservedAudioEndSec);
        if (CenterSec <= IslandEndSec)
        {
            continue;
        }

        BestLateDistanceSec = FMath::Min(BestLateDistanceSec, CenterSec - IslandEndSec);
    }
    return BestLateDistanceSec == std::numeric_limits<float>::max() ? 0.0f : BestLateDistanceSec * 1000.0f;
}

static float PercentileMs(std::vector<float> Values, double Percentile)
{
    if (Values.empty())
    {
        return 0.0f;
    }
    std::sort(Values.begin(), Values.end());
    const double ClampedPercentile = std::clamp(Percentile, 0.0, 1.0);
    const size_t Index = std::min(
        Values.size() - 1,
        static_cast<size_t>(std::ceil(static_cast<double>(Values.size()) * ClampedPercentile) - 1.0));
    return Values[Index];
}

static float Percentile90Ms(std::vector<float> Values)
{
    return PercentileMs(std::move(Values), 0.90);
}

static float Percentile95Ms(std::vector<float> Values)
{
    return PercentileMs(std::move(Values), 0.95);
}

static float RawPlanEventStartSec(const FOffgridAITextVisemeEvent& Event, float SpeechStartSec, float PlanDurationSec)
{
    return SpeechStartSec + FMath::Clamp(Event.StartNorm, 0.0f, 1.0f) * PlanDurationSec;
}

static float RawPlanEventEndSec(const FOffgridAITextVisemeEvent& Event, float SpeechStartSec, float PlanDurationSec)
{
    const float StartNorm = FMath::Clamp(Event.StartNorm, 0.0f, 1.0f);
    const float EndNorm = FMath::Clamp(Event.EndNorm, StartNorm, 1.0f);
    return SpeechStartSec + EndNorm * PlanDurationSec;
}

static FTimingDiagnostics ComputeTimingDiagnostics(
    const FOffgridAITextVisemePlan& Plan,
    const FWavData& Wav,
    const FOffgridAIAlignedVisemeTrack& StreamingTrack,
    const TArray<FOffgridAIStreamingSpeechIsland>& SpeechIslands,
    float AudioDurationSec)
{
    FTimingDiagnostics Diagnostics;
    const FOffgridAIAlignedVisemeTrack EvidenceTrack = StreamingTrack;

    std::map<int32, const FOffgridAIAlignedVisemeEvent*> StreamingByEventIndex;
    for (const FOffgridAIAlignedVisemeEvent& Event : StreamingTrack.Events)
    {
        StreamingByEventIndex[Event.EventIndex] = &Event;
    }

    const float RawSpeechStartSec = EvidenceTrack.SpeechStartSeconds;
    const float RawPlanDurationSec = FMath::Max(0.050f, Plan.EstimatedDurationSeconds);
    (void)SpeechIslands;
    (void)AudioDurationSec;

    for (const FOffgridAIAlignedVisemeEvent& EvidenceEvent : EvidenceTrack.Events)
    {
        const auto It = StreamingByEventIndex.find(EvidenceEvent.EventIndex);
        if (It == StreamingByEventIndex.end() || It->second == nullptr)
        {
            continue;
        }

        const FOffgridAIAlignedVisemeEvent& StreamingEvent = *It->second;
        if (!Plan.Events.IsValidIndex(EvidenceEvent.EventIndex))
        {
            continue;
        }

        const FOffgridAITextVisemeEvent& PlanEvent = Plan.Events[EvidenceEvent.EventIndex];
        const float RawTextStartSec = RawPlanEventStartSec(PlanEvent, RawSpeechStartSec, RawPlanDurationSec);
        const float RawTextEndSec = RawPlanEventEndSec(PlanEvent, RawSpeechStartSec, RawPlanDurationSec);
        const float RawTextCenterSec = 0.5f * (RawTextStartSec + RawTextEndSec);
        const float EvidenceCenterSec = EvidenceEvent.FinalRenderCenterSeconds;
        const float StreamingCenterSec = StreamingEvent.FinalRenderCenterSeconds;
        const float RawTextLeadBiasMs = (EvidenceCenterSec - RawTextCenterSec) * 1000.0f;
        const float StreamingLeadBiasMs = (EvidenceCenterSec - StreamingCenterSec) * 1000.0f;
        const float RawTextAbsErrorMs = FMath::Abs(RawTextLeadBiasMs);
        const float StreamingAbsErrorMs = FMath::Abs(StreamingLeadBiasMs);

        FTimingDiagnosticRow Row;
        Row.EventIndex = EvidenceEvent.EventIndex;
        Row.PoseID = EvidenceEvent.PoseID.ToString().ToStdString();
        Row.SourceWord = EvidenceEvent.SourceWord.ToStdString();
        Row.WordIndex = EvidenceEvent.WordIndex;
        Row.PhraseIndex = EvidenceEvent.PhraseIndex;
        Row.RawTextStartSec = RawTextStartSec;
        Row.RawTextCenterSec = RawTextCenterSec;
        Row.RawTextEndSec = RawTextEndSec;
        Row.StreamingCenterSec = StreamingCenterSec;
        Row.EvidenceStartSec = EvidenceEvent.RenderStartSeconds;
        Row.EvidenceCenterSec = EvidenceCenterSec;
        Row.EvidenceEndSec = EvidenceEvent.RenderEndSeconds;
        Row.RawTextErrorMs = RawTextAbsErrorMs;
        Row.StreamingErrorMs = StreamingAbsErrorMs;
        Row.RawTextLeadBiasMs = RawTextLeadBiasMs;
        Row.StreamingLeadBiasMs = StreamingLeadBiasMs;
        Row.EvidenceConfidence = EvidenceEvent.AlignmentConfidence;
        Row.bEvidenceFallback = EvidenceEvent.bUsedLayer1Fallback;
        Row.LandmarkClass = LandmarkClassForPose(StreamingEvent.PoseID);
        Row.bNudgeProposed = WasNudgeProposedForEvent(StreamingEvent);
        Row.bNudgeAccepted = WasNudgeAcceptedForEvent(StreamingEvent);
        Row.NudgeDeltaMs = StreamingEvent.AppliedShiftSeconds * 1000.0f;
        Row.NudgeReason = NudgeReasonForEvent(StreamingEvent);
        Diagnostics.Rows.push_back(Row);

        Diagnostics.MatchedEventCount += 1;
        Diagnostics.EvidenceAnchoredEventCount += EvidenceEvent.bUsedLayer1Fallback ? 0 : 1;
        Diagnostics.RawTextAbsErrorSumMs += RawTextAbsErrorMs;
        Diagnostics.StreamingAbsErrorSumMs += StreamingAbsErrorMs;
        Diagnostics.RawTextLeadBiasSumMs += RawTextLeadBiasMs;
        Diagnostics.StreamingLeadBiasSumMs += StreamingLeadBiasMs;
        Diagnostics.RawTextAbsErrorsMs.push_back(RawTextAbsErrorMs);
        Diagnostics.StreamingAbsErrorsMs.push_back(StreamingAbsErrorMs);

        if (!EvidenceEvent.bUsedLayer1Fallback)
        {
            Diagnostics.AnchoredMatchedEventCount += 1;
            Diagnostics.AnchoredRawTextAbsErrorSumMs += RawTextAbsErrorMs;
            Diagnostics.AnchoredStreamingAbsErrorSumMs += StreamingAbsErrorMs;
            Diagnostics.AnchoredRawTextAbsErrorsMs.push_back(RawTextAbsErrorMs);
            Diagnostics.AnchoredStreamingAbsErrorsMs.push_back(StreamingAbsErrorMs);
        }
    }

    return Diagnostics;
}

static void WriteTimingDiagnosticsCsv(const fs::path& Path, const FTimingDiagnostics& Diagnostics)
{
    std::ofstream Out(Path);
    Out << "EventIndex,PoseID,SourceWord,WordIndex,PhraseIndex,RawTextStartSec,RawTextCenterSec,RawTextEndSec,StreamingCenterSec,EvidenceStartSec,EvidenceCenterSec,EvidenceEndSec,RawTextErrorMs,StreamingErrorMs,RawTextLeadBiasMs,StreamingLeadBiasMs,EvidenceConfidence,EvidenceFallback,LandmarkClass,NudgeProposed,NudgeAccepted,NudgeDeltaMs,NudgeReason\n";
    for (const FTimingDiagnosticRow& Row : Diagnostics.Rows)
    {
        Out << Row.EventIndex << ','
            << CsvEscape(Row.PoseID) << ','
            << CsvEscape(Row.SourceWord) << ','
            << Row.WordIndex << ','
            << Row.PhraseIndex << ','
            << Row.RawTextStartSec << ','
            << Row.RawTextCenterSec << ','
            << Row.RawTextEndSec << ','
            << Row.StreamingCenterSec << ','
            << Row.EvidenceStartSec << ','
            << Row.EvidenceCenterSec << ','
            << Row.EvidenceEndSec << ','
            << Row.RawTextErrorMs << ','
            << Row.StreamingErrorMs << ','
            << Row.RawTextLeadBiasMs << ','
            << Row.StreamingLeadBiasMs << ','
            << Row.EvidenceConfidence << ','
            << (Row.bEvidenceFallback ? 1 : 0) << ','
            << CsvEscape(Row.LandmarkClass) << ','
            << (Row.bNudgeProposed ? 1 : 0) << ','
            << (Row.bNudgeAccepted ? 1 : 0) << ','
            << Row.NudgeDeltaMs << ','
            << CsvEscape(Row.NudgeReason) << "\n";
    }
}


static void WriteEventSubmissionSummaryCsv(
    const fs::path& Path,
    const FOffgridAITextVisemePlan& Plan,
    const FOffgridAIAlignedVisemeTrack& Track,
    const FSubmittedPoseDiagnostics& PoseDiagnostics)
{
    std::map<int32, const FOffgridAIAlignedVisemeEvent*> CommittedByEventIndex;
    for (const FOffgridAIAlignedVisemeEvent& E : Track.Events)
    {
        CommittedByEventIndex[E.EventIndex] = &E;
    }

    std::ofstream Out(Path);
    Out << "EventIndex,PoseID,SourceWord,WordIndex,PhraseIndex,TextIslandIndex,AudioIslandIndex,LandmarkClass,PlannedStrength,CommittedCenterSec,PeakSubmittedWeight,PeakPlaybackSec,PeakWindowStartSec,PeakWindowEndSec,PeakWindowWidthSec,NeverSubmitted,WeakPeak,NudgeProposed,NudgeAccepted,NudgeDeltaMs,NudgeReason\n";
    constexpr float WeakPeakThreshold = 0.35f;
    for (int32 EventIndex = 0; EventIndex < Plan.Events.Num(); ++EventIndex)
    {
        const FOffgridAITextVisemeEvent& PlanEvent = Plan.Events[EventIndex];
        const auto CommitIt = CommittedByEventIndex.find(EventIndex);
        const FOffgridAIAlignedVisemeEvent* Committed = CommitIt != CommittedByEventIndex.end() ? CommitIt->second : nullptr;
        const auto PeakIt = PoseDiagnostics.Peaks.find(EventIndex);
        const FPeakInfo* Peak = PeakIt != PoseDiagnostics.Peaks.end() ? &PeakIt->second : nullptr;
        const float PeakWeight = Peak ? Peak->PeakWeight : -1.0f;
        const bool bNeverSubmitted = !Peak || Peak->PeakWeight < 0.0f;
        const bool bWeakPeak = !bNeverSubmitted && PeakWeight < WeakPeakThreshold;
        const FName PoseID = Committed ? Committed->PoseID : PlanEvent.PoseID;
        Out << EventIndex << ','
            << CsvEscape(PoseID.ToString().ToStdString()) << ','
            << CsvEscape((Committed ? Committed->SourceWord : PlanEvent.SourceText).ToStdString()) << ','
            << (Committed ? Committed->WordIndex : PlanEvent.WordIndex) << ','
            << (Committed ? Committed->PhraseIndex : PlanEvent.PhraseIndex) << ','
            << (Committed ? Committed->TextIslandIndex : INDEX_NONE) << ','
            << (Committed ? Committed->AudioIslandIndex : INDEX_NONE) << ','
            << CsvEscape(LandmarkClassForPose(PoseID)) << ','
            << PlanEvent.Strength << ','
            << (Committed ? Committed->FinalRenderCenterSeconds : 0.0f) << ','
            << PeakWeight << ','
            << (Peak ? PeakWindowWeightedCenterTime(*Peak) : 0.0f) << ','
            << (Peak ? Peak->PeakWindowStart : 0.0f) << ','
            << (Peak ? Peak->PeakWindowEnd : 0.0f) << ','
            << (Peak ? PeakWindowWidthSec(*Peak) : 0.0f) << ','
            << (bNeverSubmitted ? 1 : 0) << ','
            << (bWeakPeak ? 1 : 0) << ','
            << (Committed && WasNudgeProposedForEvent(*Committed) ? 1 : 0) << ','
            << (Committed && WasNudgeAcceptedForEvent(*Committed) ? 1 : 0) << ','
            << (Committed ? Committed->AppliedShiftSeconds * 1000.0f : 0.0f) << ','
            << CsvEscape(Committed ? NudgeReasonForEvent(*Committed) : "not_committed") << "\n";
    }
}

static void WriteNudgeDiagnosticsCsv(const fs::path& Path, const FOffgridAIAlignedVisemeTrack& Track)
{
    std::ofstream Out(Path);
    Out << "EventIndex,PoseID,SourceWord,WordIndex,PhraseIndex,TextIslandIndex,AudioIslandIndex,LandmarkClass,PlannedCenterSec,FinalCenterSec,AlignmentConfidence,RawShiftMs,AppliedShiftMs,ShiftCapMs,CenterOrderRepaired,CenterOrderRepairMs\n";
    for (const FOffgridAIAlignedVisemeEvent& E : Track.Events)
    {
        Out << E.EventIndex << ','
            << CsvEscape(E.PoseID.ToString().ToStdString()) << ','
            << CsvEscape(E.SourceWord.ToStdString()) << ','
            << E.WordIndex << ','
            << E.PhraseIndex << ','
            << E.TextIslandIndex << ','
            << E.AudioIslandIndex << ','
            << CsvEscape(LandmarkClassForPose(E.PoseID)) << ','
            << E.TextDiagnosticCenterSeconds << ','
            << E.FinalRenderCenterSeconds << ','
            << E.AlignmentConfidence << ','
            << (E.RawShiftSeconds * 1000.0f) << ','
            << (E.AppliedShiftSeconds * 1000.0f) << ','
            << (E.ShiftCapSeconds * 1000.0f) << ','
            << (E.bCenterOrderRepaired ? 1 : 0) << ','
            << (E.CenterOrderRepairSeconds * 1000.0f) << "\n";
    }
}

static float Alpha32aLegalMinCenterSec(const FOffgridAIAlignedVisemeTrack& Track, int32 TrackIndex, float GuardSec)
{
    const FOffgridAIAlignedVisemeEvent& Event = Track.Events[TrackIndex];
    float MinSec = Event.IslandAudioStartSeconds + GuardSec;
    for (int32 J = TrackIndex - 1; J >= 0; --J)
    {
        if (Track.Events[J].TextIslandIndex == Event.TextIslandIndex)
        {
            MinSec = FMath::Max(MinSec, Track.Events[J].FinalRenderCenterSeconds + GuardSec);
            break;
        }
    }
    return MinSec;
}

static float Alpha32aLegalMaxCenterSec(const FOffgridAIAlignedVisemeTrack& Track, int32 TrackIndex, float GuardSec)
{
    const FOffgridAIAlignedVisemeEvent& Event = Track.Events[TrackIndex];
    float MaxSec = Event.IslandAudioEndSeconds - GuardSec;
    for (int32 J = TrackIndex + 1; J < Track.Events.Num(); ++J)
    {
        if (Track.Events[J].TextIslandIndex == Event.TextIslandIndex)
        {
            MaxSec = FMath::Min(MaxSec, Track.Events[J].FinalRenderCenterSeconds - GuardSec);
            break;
        }
    }
    return MaxSec;
}

static void ApplyAlpha32aClippedEventOnlyNudges(
    const FOffgridAITextVisemePlan& Plan,
    const FWavData& Wav,
    FOffgridAIAlignedVisemeTrack& Track)
{
    const FOffgridAIAlignedVisemeTrack EvidenceTrack = Track;

    std::map<int32, const FOffgridAIAlignedVisemeEvent*> EvidenceByEventIndex;
    for (const FOffgridAIAlignedVisemeEvent& E : EvidenceTrack.Events)
    {
        EvidenceByEventIndex[E.EventIndex] = &E;
    }

    constexpr float RequiredConfidence = 0.09f;
    constexpr float GuardSec = 0.001f;
    for (int32 I = 0; I < Track.Events.Num(); ++I)
    {
        FOffgridAIAlignedVisemeEvent& Event = Track.Events[I];
        const std::string NudgeClass = LandmarkClassForPose(Event.PoseID);
        const bool bHardEventNudgeEligible = IsAlpha31PrimaryAuditPose(Event.PoseID);
        const bool bSoftTimelineAnchorEligible = IsAu24SoftTimelineAnchorClass(NudgeClass);
        Event.bAudioNudgeEligible = bHardEventNudgeEligible || bSoftTimelineAnchorEligible;
        Event.bAudioNudgeSearchPerformed = Event.bAudioNudgeEligible;
        Event.AudioNudgeSearchMode = bSoftTimelineAnchorEligible && !bHardEventNudgeEligible
            ? FName(TEXT("au24_soft_timeline_anchor_probe"))
            : FName(TEXT("alpha32a_offline_clipped_event_only"));
        Event.AudioNudgeScheduledCenterSeconds = Event.FinalRenderCenterSeconds;
        Event.AudioNudgeRequiredConfidence = bSoftTimelineAnchorEligible
            ? Au24CalibratedTimelineAnchorConfidenceForClass(NudgeClass)
            : RequiredConfidence;
        Event.AudioNudgeMaxCorrectionSeconds = FMath::Max(FMath::Abs(Alpha31ClampMinMs(Event.PoseID)), FMath::Abs(Alpha31ClampMaxMs(Event.PoseID))) * 0.001f;

        if (!Event.bAudioNudgeEligible)
        {
            Event.AudioNudgeRejectReason = FName(TEXT("not_alpha32a_or_au24_anchor_pose"));
            continue;
        }

        const auto EvidenceIt = EvidenceByEventIndex.find(Event.EventIndex);
        const FOffgridAIAlignedVisemeEvent* Evidence = EvidenceIt != EvidenceByEventIndex.end() ? EvidenceIt->second : nullptr;
        const bool bHasEvidence = Evidence != nullptr && Evidence->AlignmentConfidence > 0.0f && !Evidence->bUsedLayer1Fallback;
        if (!bHasEvidence)
        {
            Event.AudioNudgeRejectReason = FName(TEXT("no_offline_audio_evidence"));
            continue;
        }

        Event.AudioNudgeCandidateConfidence = Evidence->AlignmentConfidence;
        const float RequiredConfidenceForEvent = Event.AudioNudgeRequiredConfidence;
        if (Evidence->AlignmentConfidence < RequiredConfidenceForEvent)
        {
            Event.AudioNudgeRejectReason = bSoftTimelineAnchorEligible
                ? FName(TEXT("low_au24_soft_anchor_confidence"))
                : FName(TEXT("low_offline_audio_confidence"));
            continue;
        }

        const float OriginalCenterSec = Event.FinalRenderCenterSeconds;
        const float TargetCenterSec = Evidence->FinalRenderCenterSeconds - Alpha31VisualLeadSeconds(Event.PoseID);
        const float RawShiftSec = TargetCenterSec - OriginalCenterSec;
        const float ClassClampedShiftSec = FMath::Clamp(
            RawShiftSec,
            Alpha31ClampMinMs(Event.PoseID) * 0.001f,
            Alpha31ClampMaxMs(Event.PoseID) * 0.001f);
        const float ClassClampedTargetSec = OriginalCenterSec + ClassClampedShiftSec;

        const float LegalMinSec = Alpha32aLegalMinCenterSec(Track, I, GuardSec);
        const float LegalMaxSec = Alpha32aLegalMaxCenterSec(Track, I, GuardSec);
        if (LegalMaxSec < LegalMinSec)
        {
            Event.AudioNudgeRejectReason = FName(TEXT("empty_legal_envelope"));
            continue;
        }

        const float AppliedCenterSec = FMath::Clamp(ClassClampedTargetSec, LegalMinSec, LegalMaxSec);
        const float AppliedShiftSec = AppliedCenterSec - OriginalCenterSec;

        Event.RawShiftSeconds = RawShiftSec;
        Event.AppliedShiftSeconds = AppliedShiftSec;
        Event.ShiftCapSeconds = ClassClampedShiftSec;
        Event.AudioNudgeCandidateRawCenterSeconds = TargetCenterSec;
        Event.AudioNudgeCandidateAppliedCenterSeconds = AppliedCenterSec;
        Event.AudioNudgeCandidateRawShiftSeconds = RawShiftSec;
        Event.AudioNudgeCandidateAppliedShiftSeconds = AppliedShiftSec;
        Event.AudioNudgeSearchStartSeconds = LegalMinSec;
        Event.AudioNudgeSearchEndSeconds = LegalMaxSec;
        Event.AudioNudgeObservedEndSeconds = Wav.SampleRate > 0 ? static_cast<float>(Wav.SamplesInterleaved.size() / std::max(1, Wav.Channels)) / static_cast<float>(Wav.SampleRate) : 0.0f;
        Event.AudioNudgePredictedLeadSeconds = Alpha31VisualLeadSeconds(Event.PoseID);

        if (bSoftTimelineAnchorEligible && !bHardEventNudgeEligible)
        {
            Event.AudioNudgeRejectReason = FName(TEXT("au24_soft_timeline_anchor_candidate"));
            continue;
        }

        if (FMath::Abs(AppliedShiftSec) < 0.0005f)
        {
            Event.AudioNudgeRejectReason = FName(TEXT("evidence_no_shift_after_clip"));
            continue;
        }

        Event.FinalRenderCenterSeconds = AppliedCenterSec;
        Event.RenderStartSeconds += AppliedShiftSec;
        Event.RenderEndSeconds += AppliedShiftSec;
        Event.CommitLeadSeconds += AppliedShiftSec;
        Event.bAudioNudgeAccepted = true;
        Event.AudioNudgeRejectReason = FName(TEXT("accepted_alpha32a_clipped_event_only"));

    }
}


struct FAu22PiecewiseWarpRow
{
    int32 EventIndex = INDEX_NONE;
    int32 TextIslandIndex = INDEX_NONE;
    FString PoseID;
    FString SourceWord;
    float OriginalCenterSec = 0.0f;
    float BeforeWarpCenterSec = 0.0f;
    float WarpedCenterSec = 0.0f;
    float WarpShiftMs = 0.0f;
    float PrevAnchorEventIndex = -1.0f;
    float NextAnchorEventIndex = -1.0f;
    FString WarpMode;
    bool bAnchor = false;
    FString AnchorClass;
    FString AnchorKind;
    float AnchorConfidence = 0.0f;
    float AnchorThreshold = 0.0f;
    float AnchorBlend = 1.0f;
    float AnchorMaxShiftSec = 0.0f;
};

static bool IsAu24CalibratedTimelineAnchor(const FOffgridAIAlignedVisemeEvent& Event)
{
    const std::string Class = LandmarkClassForPose(Event.PoseID);
    if (!IsAu24TimelineAnchorClass(Class))
    {
        return false;
    }

    const float Required = Au24CalibratedTimelineAnchorConfidenceForClass(Class);
    constexpr float MinAnchorShiftSec = 0.006f;
    const float CandidateShiftSec = Event.AudioNudgeCandidateRawCenterSeconds > 0.0f
        ? Event.AudioNudgeCandidateRawCenterSeconds - Event.AudioNudgeScheduledCenterSeconds
        : Event.AudioNudgeCandidateAppliedShiftSeconds;

    if (Event.AudioNudgeCandidateConfidence < Required ||
        FMath::Abs(CandidateShiftSec) < MinAnchorShiftSec)
    {
        return false;
    }

    // Hard anchors must have survived the existing event-only nudge safety
    // envelope.  Soft rounded/front vowel anchors may guide the surrounding
    // timeline even though the event itself was not hard-nudged.
    if (Class == "MBP" || Class == "FV" || Class == "WOO_GLIDE")
    {
        return WasNudgeAcceptedForEvent(Event);
    }
    return IsAu24SoftTimelineAnchorClass(Class) && Event.AudioNudgeCandidateRawCenterSeconds > 0.0f;
}

static FString Au24AnchorKindForEvent(const FOffgridAIAlignedVisemeEvent& Event)
{
    const std::string Class = LandmarkClassForPose(Event.PoseID);
    if (Class == "MBP" || Class == "FV" || Class == "WOO_GLIDE")
    {
        return TEXT("hard_existing_nudge_anchor");
    }
    if (IsAu24SoftTimelineAnchorClass(Class))
    {
        return TEXT("soft_vowel_timeline_anchor");
    }
    return TEXT("not_timeline_anchor");
}

static float Au26TimelineAnchorBlendForClass(const std::string& Class, float Confidence)
{
    // AU26: hard landmarks remain exact anchors; vowel anchors are admitted as
    // soft timing evidence.  Their target is blended toward the raw offline
    // evidence so a confident but slightly wrong vowel cannot drag an entire
    // local timeline segment as aggressively as MBP/FV/WOO.
    if (Class == "MBP" || Class == "FV" || Class == "WOO_GLIDE")
    {
        return 1.0f;
    }
    if (Class == "ROUNDED_VOWEL" || Class == "FRONT_VOWEL")
    {
        const float Required = Au24CalibratedTimelineAnchorConfidenceForClass(Class);
        const float Extra = FMath::Clamp((Confidence - Required) / FMath::Max(0.001f, 1.0f - Required), 0.0f, 1.0f);
        return 0.55f + 0.30f * Extra;
    }
    return 0.0f;
}

static float Au26TimelineAnchorMaxShiftSecForClass(const std::string& Class)
{
    if (Class == "MBP" || Class == "FV" || Class == "WOO_GLIDE")
    {
        return 0.220f;
    }
    if (Class == "ROUNDED_VOWEL")
    {
        return 0.145f;
    }
    if (Class == "FRONT_VOWEL")
    {
        return 0.125f;
    }
    return 0.0f;
}

static void ApplyAu26PiecewiseTimelineWarpFromAudioAnchors(
    const FOffgridAIAlignedVisemeTrack& PreNudgeTrack,
    FOffgridAIAlignedVisemeTrack& Track,
    std::vector<FAu22PiecewiseWarpRow>& OutRows)
{
    struct FOriginalEvent
    {
        float CenterSec = 0.0f;
        float RenderStartSec = 0.0f;
        float RenderEndSec = 0.0f;
        bool bValid = false;
    };
    struct FAnchor
    {
        int32 TrackIndex = INDEX_NONE;
        int32 EventIndex = INDEX_NONE;
        int32 TextIslandIndex = INDEX_NONE;
        float PlannedSec = 0.0f;
        float TargetSec = 0.0f;
        float Confidence = 1.0f;
        float Threshold = 0.0f;
        FString AnchorClass;
        FString AnchorKind;
        float Blend = 1.0f;
        float MaxShiftSec = 0.0f;
        FString Reason;
    };

    std::map<int32, FOriginalEvent> OriginalByEventIndex;
    for (const FOffgridAIAlignedVisemeEvent& E : PreNudgeTrack.Events)
    {
        FOriginalEvent O;
        O.CenterSec = E.FinalRenderCenterSeconds;
        O.RenderStartSec = E.RenderStartSeconds;
        O.RenderEndSec = E.RenderEndSeconds;
        O.bValid = true;
        OriginalByEventIndex[E.EventIndex] = O;
    }

    std::map<int32, std::vector<int32>> TrackIndicesByIsland;
    for (int32 I = 0; I < Track.Events.Num(); ++I)
    {
        TrackIndicesByIsland[Track.Events[I].TextIslandIndex].push_back(I);
    }

    constexpr float MinAnchorGapSec = 0.030f;
    constexpr float MinTargetGapSec = 0.012f;
    constexpr float MaxNonAnchorShiftSec = 0.180f;
    constexpr float MaxExtrapolatedScale = 1.18f;
    constexpr float MinExtrapolatedScale = 0.78f;
    constexpr float MinCommittedGapSec = 0.006f;

    for (const auto& Pair : TrackIndicesByIsland)
    {
        const std::vector<int32>& Indices = Pair.second;
        if (Indices.size() < 2)
        {
            continue;
        }

        std::vector<FAnchor> Anchors;
        const FOffgridAIAlignedVisemeEvent& FirstEvent = Track.Events[Indices.front()];
        FAnchor StartAnchor;
        StartAnchor.TrackIndex = INDEX_NONE;
        StartAnchor.EventIndex = INDEX_NONE;
        StartAnchor.TextIslandIndex = Pair.first;
        StartAnchor.PlannedSec = FirstEvent.IslandAudioStartSeconds;
        StartAnchor.TargetSec = FirstEvent.IslandAudioStartSeconds;
        StartAnchor.Confidence = 1.0f;
        StartAnchor.Threshold = 1.0f;
        StartAnchor.AnchorClass = TEXT("ISLAND_START");
        StartAnchor.AnchorKind = TEXT("island_start");
        StartAnchor.Blend = 1.0f;
        StartAnchor.MaxShiftSec = 0.0f;
        StartAnchor.Reason = TEXT("island_start");
        Anchors.push_back(StartAnchor);

        for (int32 TrackIndex : Indices)
        {
            const FOffgridAIAlignedVisemeEvent& E = Track.Events[TrackIndex];
            if (!IsAu24CalibratedTimelineAnchor(E))
            {
                continue;
            }
            const auto OriginalIt = OriginalByEventIndex.find(E.EventIndex);
            if (OriginalIt == OriginalByEventIndex.end() || !OriginalIt->second.bValid)
            {
                continue;
            }
            FAnchor A;
            A.TrackIndex = TrackIndex;
            A.EventIndex = E.EventIndex;
            A.TextIslandIndex = E.TextIslandIndex;
            const std::string AnchorClass = LandmarkClassForPose(E.PoseID);
            const float RawTargetSec = E.AudioNudgeCandidateRawCenterSeconds > 0.0f ? E.AudioNudgeCandidateRawCenterSeconds : E.FinalRenderCenterSeconds;
            const float Blend = Au26TimelineAnchorBlendForClass(AnchorClass, E.AudioNudgeCandidateConfidence);
            const float MaxShiftSec = Au26TimelineAnchorMaxShiftSecForClass(AnchorClass);
            const float BlendedShiftSec = FMath::Clamp((RawTargetSec - OriginalIt->second.CenterSec) * Blend, -MaxShiftSec, MaxShiftSec);

            A.PlannedSec = OriginalIt->second.CenterSec;
            A.TargetSec = A.PlannedSec + BlendedShiftSec;
            A.Confidence = E.AudioNudgeCandidateConfidence;
            A.Threshold = Au24CalibratedTimelineAnchorConfidenceForClass(AnchorClass);
            A.AnchorClass = FString(AnchorClass.c_str());
            A.AnchorKind = Au24AnchorKindForEvent(E);
            A.Blend = Blend;
            A.MaxShiftSec = MaxShiftSec;
            A.Reason = E.AudioNudgeRejectReason.ToString();
            Anchors.push_back(A);
        }

        std::sort(Anchors.begin(), Anchors.end(), [](const FAnchor& A, const FAnchor& B)
        {
            if (!FMath::IsNearlyEqual(A.PlannedSec, B.PlannedSec, 0.0005f))
            {
                return A.PlannedSec < B.PlannedSec;
            }
            return A.TargetSec < B.TargetSec;
        });

        std::vector<FAnchor> Filtered;
        for (const FAnchor& A : Anchors)
        {
            if (!Filtered.empty())
            {
                const FAnchor& Prev = Filtered.back();
                if (A.PlannedSec < Prev.PlannedSec + MinAnchorGapSec)
                {
                    continue;
                }
                if (A.TargetSec < Prev.TargetSec + MinTargetGapSec)
                {
                    continue;
                }
            }
            Filtered.push_back(A);
        }
        Anchors.swap(Filtered);
        if (Anchors.size() < 2)
        {
            continue;
        }

        for (int32 TrackIndex : Indices)
        {
            FOffgridAIAlignedVisemeEvent& E = Track.Events[TrackIndex];
            const auto OriginalIt = OriginalByEventIndex.find(E.EventIndex);
            if (OriginalIt == OriginalByEventIndex.end() || !OriginalIt->second.bValid)
            {
                continue;
            }
            const FOriginalEvent& Original = OriginalIt->second;
            const float OriginalCenterSec = Original.CenterSec;
            const float BeforeWarpCenterSec = E.FinalRenderCenterSeconds;

            int32 ExactAnchorIndex = INDEX_NONE;
            for (int32 AnchorI = 0; AnchorI < static_cast<int32>(Anchors.size()); ++AnchorI)
            {
                if (Anchors[AnchorI].EventIndex == E.EventIndex && Anchors[AnchorI].EventIndex != INDEX_NONE)
                {
                    ExactAnchorIndex = AnchorI;
                    break;
                }
            }

            float DesiredCenterSec = BeforeWarpCenterSec;
            int32 PrevAnchorI = 0;
            int32 NextAnchorI = INDEX_NONE;
            FString Mode = TEXT("unchanged");
            const bool bIsAnchor = ExactAnchorIndex != INDEX_NONE;

            if (bIsAnchor)
            {
                DesiredCenterSec = Anchors[ExactAnchorIndex].TargetSec;
                PrevAnchorI = ExactAnchorIndex;
                NextAnchorI = ExactAnchorIndex;
                Mode = TEXT("anchor_kept");
            }
            else
            {
                for (int32 AnchorI = 1; AnchorI < static_cast<int32>(Anchors.size()); ++AnchorI)
                {
                    if (OriginalCenterSec <= Anchors[AnchorI].PlannedSec)
                    {
                        PrevAnchorI = AnchorI - 1;
                        NextAnchorI = AnchorI;
                        break;
                    }
                }

                if (NextAnchorI != INDEX_NONE)
                {
                    const FAnchor& A = Anchors[PrevAnchorI];
                    const FAnchor& B = Anchors[NextAnchorI];
                    const float Den = FMath::Max(0.001f, B.PlannedSec - A.PlannedSec);
                    const float T = FMath::Clamp((OriginalCenterSec - A.PlannedSec) / Den, 0.0f, 1.0f);
                    DesiredCenterSec = A.TargetSec + T * (B.TargetSec - A.TargetSec);
                    Mode = TEXT("interpolated_between_anchors");
                }
                else
                {
                    PrevAnchorI = static_cast<int32>(Anchors.size()) - 1;
                    const FAnchor& Last = Anchors[PrevAnchorI];
                    const FAnchor& Prev = Anchors[FMath::Max(0, PrevAnchorI - 1)];
                    float Scale = 1.0f;
                    if (Last.PlannedSec > Prev.PlannedSec + 0.001f)
                    {
                        Scale = (Last.TargetSec - Prev.TargetSec) / (Last.PlannedSec - Prev.PlannedSec);
                    }
                    Scale = FMath::Clamp(Scale, MinExtrapolatedScale, MaxExtrapolatedScale);
                    DesiredCenterSec = Last.TargetSec + (OriginalCenterSec - Last.PlannedSec) * Scale;
                    Mode = TEXT("tail_extrapolated_from_last_anchor_pair");
                }
            }

            const float RawWarpShiftSec = DesiredCenterSec - OriginalCenterSec;
            const float CappedWarpShiftSec = bIsAnchor
                ? (DesiredCenterSec - OriginalCenterSec)
                : FMath::Clamp(RawWarpShiftSec, -MaxNonAnchorShiftSec, MaxNonAnchorShiftSec);
            float NewCenterSec = OriginalCenterSec + CappedWarpShiftSec;
            const float DeltaFromBeforeSec = NewCenterSec - BeforeWarpCenterSec;
            if (FMath::Abs(DeltaFromBeforeSec) > 0.0005f)
            {
                E.FinalRenderCenterSeconds += DeltaFromBeforeSec;
                E.RenderStartSeconds += DeltaFromBeforeSec;
                E.RenderEndSeconds += DeltaFromBeforeSec;
                E.CommitLeadSeconds += DeltaFromBeforeSec;
                if (!bIsAnchor)
                {
                    E.AudioNudgeSearchMode = FName(TEXT("au26_piecewise_timeline_warp"));
                }
            }

            FAu22PiecewiseWarpRow Row;
            Row.EventIndex = E.EventIndex;
            Row.TextIslandIndex = E.TextIslandIndex;
            Row.PoseID = E.PoseID.ToString();
            Row.SourceWord = E.SourceWord;
            Row.OriginalCenterSec = OriginalCenterSec;
            Row.BeforeWarpCenterSec = BeforeWarpCenterSec;
            Row.WarpedCenterSec = E.FinalRenderCenterSeconds;
            Row.WarpShiftMs = (E.FinalRenderCenterSeconds - OriginalCenterSec) * 1000.0f;
            Row.PrevAnchorEventIndex = Anchors[PrevAnchorI].EventIndex;
            Row.NextAnchorEventIndex = NextAnchorI != INDEX_NONE ? Anchors[NextAnchorI].EventIndex : INDEX_NONE;
            Row.WarpMode = Mode;
            Row.bAnchor = bIsAnchor;
            if (bIsAnchor)
            {
                Row.AnchorClass = Anchors[ExactAnchorIndex].AnchorClass;
                Row.AnchorKind = Anchors[ExactAnchorIndex].AnchorKind;
                Row.AnchorConfidence = Anchors[ExactAnchorIndex].Confidence;
                Row.AnchorThreshold = Anchors[ExactAnchorIndex].Threshold;
                Row.AnchorBlend = Anchors[ExactAnchorIndex].Blend;
                Row.AnchorMaxShiftSec = Anchors[ExactAnchorIndex].MaxShiftSec;
            }
            OutRows.push_back(Row);
        }

        bool bHasPreviousCenter = false;
        float PreviousCenterSec = 0.0f;
        for (int32 TrackIndex : Indices)
        {
            FOffgridAIAlignedVisemeEvent& E = Track.Events[TrackIndex];
            if (bHasPreviousCenter && E.FinalRenderCenterSeconds < PreviousCenterSec + MinCommittedGapSec)
            {
                const float DeltaSec = PreviousCenterSec + MinCommittedGapSec - E.FinalRenderCenterSeconds;
                E.FinalRenderCenterSeconds += DeltaSec;
                E.RenderStartSeconds += DeltaSec;
                E.RenderEndSeconds += DeltaSec;
                E.CommitLeadSeconds += DeltaSec;
                E.CenterOrderRepairSeconds += DeltaSec;
                E.bCenterOrderRepaired = true;
            }
            PreviousCenterSec = E.FinalRenderCenterSeconds;
            bHasPreviousCenter = true;
        }
    }
}


static void RepairGlobalCommittedCenterOrder(FOffgridAIAlignedVisemeTrack& Track, float MinGapSec)
{
    bool bHasPreviousCenter = false;
    float PreviousCenterSec = 0.0f;
    for (FOffgridAIAlignedVisemeEvent& E : Track.Events)
    {
        if (bHasPreviousCenter && E.FinalRenderCenterSeconds < PreviousCenterSec + MinGapSec)
        {
            const float DeltaSec = PreviousCenterSec + MinGapSec - E.FinalRenderCenterSeconds;
            E.FinalRenderCenterSeconds += DeltaSec;
            E.RenderStartSeconds += DeltaSec;
            E.RenderEndSeconds += DeltaSec;
            E.CommitLeadSeconds += DeltaSec;
            E.CenterOrderRepairSeconds += DeltaSec;
            E.bCenterOrderRepaired = true;
            if (E.AudioNudgeSearchMode == NAME_None || E.AudioNudgeSearchMode == FName(TEXT("au26_piecewise_timeline_warp")))
            {
                E.AudioNudgeSearchMode = FName(TEXT("au26b_global_center_order_repair"));
            }
        }
        PreviousCenterSec = E.FinalRenderCenterSeconds;
        bHasPreviousCenter = true;
    }
}

static void WriteAu22PiecewiseTimelineWarpCsv(const fs::path& Path, const std::vector<FAu22PiecewiseWarpRow>& Rows)
{
    std::ofstream Out(Path);
    Out << "EventIndex,TextIslandIndex,PoseID,SourceWord,OriginalCenterSec,BeforeWarpCenterSec,WarpedCenterSec,WarpShiftMs,PrevAnchorEventIndex,NextAnchorEventIndex,WarpMode,IsAnchor,AnchorClass,AnchorKind,AnchorConfidence,AnchorThreshold,AnchorBlend,AnchorMaxShiftSec\n";
    for (const FAu22PiecewiseWarpRow& R : Rows)
    {
        Out << R.EventIndex << ','
            << R.TextIslandIndex << ','
            << CsvEscape(R.PoseID.ToStdString()) << ','
            << CsvEscape(R.SourceWord.ToStdString()) << ','
            << R.OriginalCenterSec << ','
            << R.BeforeWarpCenterSec << ','
            << R.WarpedCenterSec << ','
            << R.WarpShiftMs << ','
            << R.PrevAnchorEventIndex << ','
            << R.NextAnchorEventIndex << ','
            << CsvEscape(R.WarpMode.ToStdString()) << ','
            << (R.bAnchor ? 1 : 0) << ','
            << CsvEscape(R.AnchorClass.ToStdString()) << ','
            << CsvEscape(R.AnchorKind.ToStdString()) << ','
            << R.AnchorConfidence << ','
            << R.AnchorThreshold << ','
            << R.AnchorBlend << ','
            << R.AnchorMaxShiftSec << "\n";
    }
}


static void WriteAlpha31NudgeOpportunityCsv(
    const fs::path& Path,
    const FOffgridAITextVisemePlan& Plan,
    const FWavData& Wav,
    const FOffgridAIAlignedVisemeTrack& StreamingTrack,
    const FSubmittedPoseDiagnostics& PoseDiagnostics)
{
    const FOffgridAIAlignedVisemeTrack EvidenceTrack = StreamingTrack;

    std::map<int32, const FOffgridAIAlignedVisemeEvent*> StreamingByEventIndex;
    for (const FOffgridAIAlignedVisemeEvent& E : StreamingTrack.Events)
    {
        StreamingByEventIndex[E.EventIndex] = &E;
    }

    std::map<int32, const FOffgridAIAlignedVisemeEvent*> EvidenceByEventIndex;
    for (const FOffgridAIAlignedVisemeEvent& E : EvidenceTrack.Events)
    {
        EvidenceByEventIndex[E.EventIndex] = &E;
    }

    std::ofstream Out(Path);
    Out << "EventIndex,PoseID,SourceWord,WordIndex,PhraseIndex,TextIslandIndex,AudioIslandIndex,LandmarkClass,CommittedCenterSec,FacePeakSec,FacePeakWeight,OfflineEvidenceCenterSec,TargetVisualCenterSec,CommittedToTargetErrorMs,FacePeakToTargetErrorMs,FacePeakMinusCommittedMs,RawOpportunityShiftMs,ClampedOpportunityShiftMs,HypotheticalCenterSec,ClassClampMinMs,ClassClampMaxMs,OfflineEvidenceConfidence,HasOfflineEvidence,StrongOfflineEvidence,HypotheticallySafe,RejectReason,WouldCrossPrevEvent,WouldCrossNextEvent,WouldLeaveIslandBounds,WouldExceedClassClamp,PrevCommittedCenterSec,NextCommittedCenterSec,IslandStartSec,IslandEndSec,FinalCenterWouldChange\n";

    for (int32 I = 0; I < StreamingTrack.Events.Num(); ++I)
    {
        const FOffgridAIAlignedVisemeEvent& S = StreamingTrack.Events[I];
        if (!IsAlpha31PrimaryAuditPose(S.PoseID))
        {
            continue;
        }

        const auto EvidenceIt = EvidenceByEventIndex.find(S.EventIndex);
        const FOffgridAIAlignedVisemeEvent* E = EvidenceIt != EvidenceByEventIndex.end() ? EvidenceIt->second : nullptr;
        const bool bHasEvidence = E != nullptr && E->AlignmentConfidence > 0.0f && !E->bUsedLayer1Fallback;
        const float EvidenceCenterSec = E ? E->FinalRenderCenterSeconds : 0.0f;
        const float TargetVisualCenterSec = bHasEvidence ? EvidenceCenterSec - Alpha31VisualLeadSeconds(S.PoseID) : S.FinalRenderCenterSeconds;
        const auto PeakIt = PoseDiagnostics.Peaks.find(S.EventIndex);
        const FPeakInfo* Peak = PeakIt != PoseDiagnostics.Peaks.end() ? &PeakIt->second : nullptr;
        const bool bHasFacePeak = Peak != nullptr && Peak->PeakWeight >= 0.0f;
        const float FacePeakSec = bHasFacePeak ? PeakWindowWeightedCenterTime(*Peak) : 0.0f;
        const float FacePeakWeight = bHasFacePeak ? Peak->PeakWeight : -1.0f;
        const float CommittedToTargetErrorMs = bHasEvidence ? (S.FinalRenderCenterSeconds - TargetVisualCenterSec) * 1000.0f : 0.0f;
        const float FacePeakToTargetErrorMs = (bHasEvidence && bHasFacePeak) ? (FacePeakSec - TargetVisualCenterSec) * 1000.0f : 0.0f;
        const float FacePeakMinusCommittedMs = bHasFacePeak ? (FacePeakSec - S.FinalRenderCenterSeconds) * 1000.0f : 0.0f;
        const float RawShiftMs = (TargetVisualCenterSec - S.FinalRenderCenterSeconds) * 1000.0f;
        const float ClampMinMs = Alpha31ClampMinMs(S.PoseID);
        const float ClampMaxMs = Alpha31ClampMaxMs(S.PoseID);
        const float ClampedShiftMs = FMath::Clamp(RawShiftMs, ClampMinMs, ClampMaxMs);
        const float HypotheticalCenterSec = S.FinalRenderCenterSeconds + ClampedShiftMs * 0.001f;
        const bool bExceedsClamp = RawShiftMs < ClampMinMs - 0.001f || RawShiftMs > ClampMaxMs + 0.001f;

        float PrevCenterSec = 0.0f;
        float NextCenterSec = 0.0f;
        bool bHasPrev = false;
        bool bHasNext = false;
        for (int32 J = I - 1; J >= 0; --J)
        {
            if (StreamingTrack.Events[J].TextIslandIndex == S.TextIslandIndex)
            {
                PrevCenterSec = StreamingTrack.Events[J].FinalRenderCenterSeconds;
                bHasPrev = true;
                break;
            }
        }
        for (int32 J = I + 1; J < StreamingTrack.Events.Num(); ++J)
        {
            if (StreamingTrack.Events[J].TextIslandIndex == S.TextIslandIndex)
            {
                NextCenterSec = StreamingTrack.Events[J].FinalRenderCenterSeconds;
                bHasNext = true;
                break;
            }
        }

        const float GuardSec = 0.001f;
        const bool bWouldCrossPrev = bHasPrev && HypotheticalCenterSec <= PrevCenterSec + GuardSec;
        const bool bWouldCrossNext = bHasNext && HypotheticalCenterSec >= NextCenterSec - GuardSec;
        const float IslandStartSec = S.IslandAudioStartSeconds;
        const float IslandEndSec = S.IslandAudioEndSeconds;
        const bool bWouldLeaveIsland = HypotheticalCenterSec < IslandStartSec - GuardSec || HypotheticalCenterSec > IslandEndSec + GuardSec;
        const bool bStrongEvidence = bHasEvidence && E && E->AlignmentConfidence >= 0.09f;
        const std::string Reason = Alpha31RejectReason(
            bHasEvidence,
            bStrongEvidence,
            bWouldCrossPrev,
            bWouldCrossNext,
            bWouldLeaveIsland,
            bExceedsClamp,
            FMath::Abs(ClampedShiftMs));
        const bool bSafe = Reason == "hypothetically_safe" || Reason == "evidence_no_shift";

        Out << S.EventIndex << ','
            << CsvEscape(S.PoseID.ToString().ToStdString()) << ','
            << CsvEscape(S.SourceWord.ToStdString()) << ','
            << S.WordIndex << ','
            << S.PhraseIndex << ','
            << S.TextIslandIndex << ','
            << S.AudioIslandIndex << ','
            << CsvEscape(LandmarkClassForPose(S.PoseID)) << ','
            << S.FinalRenderCenterSeconds << ','
            << FacePeakSec << ','
            << FacePeakWeight << ','
            << EvidenceCenterSec << ','
            << TargetVisualCenterSec << ','
            << CommittedToTargetErrorMs << ','
            << FacePeakToTargetErrorMs << ','
            << FacePeakMinusCommittedMs << ','
            << RawShiftMs << ','
            << ClampedShiftMs << ','
            << HypotheticalCenterSec << ','
            << ClampMinMs << ','
            << ClampMaxMs << ','
            << (E ? E->AlignmentConfidence : 0.0f) << ','
            << (bHasEvidence ? 1 : 0) << ','
            << (bStrongEvidence ? 1 : 0) << ','
            << (bSafe ? 1 : 0) << ','
            << CsvEscape(Reason) << ','
            << (bWouldCrossPrev ? 1 : 0) << ','
            << (bWouldCrossNext ? 1 : 0) << ','
            << (bWouldLeaveIsland ? 1 : 0) << ','
            << (bExceedsClamp ? 1 : 0) << ','
            << (bHasPrev ? PrevCenterSec : 0.0f) << ','
            << (bHasNext ? NextCenterSec : 0.0f) << ','
            << IslandStartSec << ','
            << IslandEndSec << ','
            << 0 << "\n";
    }
}


static void WriteAu23VisemeAnchorFrontierCsv(
    const fs::path& Path,
    const FOffgridAITextVisemePlan& Plan,
    const FWavData& Wav,
    const FOffgridAIAlignedVisemeTrack& StreamingTrack)
{
    const FOffgridAIAlignedVisemeTrack EvidenceTrack = StreamingTrack;

    std::map<int32, const FOffgridAIAlignedVisemeEvent*> EvidenceByEventIndex;
    for (const FOffgridAIAlignedVisemeEvent& E : EvidenceTrack.Events)
    {
        EvidenceByEventIndex[E.EventIndex] = &E;
    }

    std::ofstream Out(Path);
    Out << "EventIndex,TextIslandIndex,AudioIslandIndex,PoseID,SourceWord,WordIndex,PhraseIndex,AnchorClass,CommittedCenterSec,OfflineEvidenceCenterSec,TargetVisualCenterSec,CommittedToTargetErrorMs,AbsCommittedToTargetErrorMs,RawShiftMs,OfflineEvidenceConfidence,RequiredAnchorConfidence,HasOfflineEvidence,StrongForClass,CurrentHardAnchorClass,ProposedNewHardAnchorClass,AnchorFrontierVerdict,NearestPrevEventGapMs,NearestNextEventGapMs\n";

    for (int32 I = 0; I < StreamingTrack.Events.Num(); ++I)
    {
        const FOffgridAIAlignedVisemeEvent& S = StreamingTrack.Events[I];
        if (!IsAu23AnchorFrontierPose(S.PoseID))
        {
            continue;
        }

        const std::string Class = LandmarkClassForPose(S.PoseID);
        const auto EvidenceIt = EvidenceByEventIndex.find(S.EventIndex);
        const FOffgridAIAlignedVisemeEvent* E = EvidenceIt != EvidenceByEventIndex.end() ? EvidenceIt->second : nullptr;
        const bool bHasEvidence = E != nullptr && E->AlignmentConfidence > 0.0f && !E->bUsedLayer1Fallback;
        const float EvidenceCenterSec = E ? E->FinalRenderCenterSeconds : 0.0f;
        const float TargetVisualCenterSec = bHasEvidence ? EvidenceCenterSec - Alpha31VisualLeadSeconds(S.PoseID) : S.FinalRenderCenterSeconds;
        const float ErrorMs = bHasEvidence ? (S.FinalRenderCenterSeconds - TargetVisualCenterSec) * 1000.0f : 0.0f;
        const float RawShiftMs = bHasEvidence ? (TargetVisualCenterSec - S.FinalRenderCenterSeconds) * 1000.0f : 0.0f;
        const float Confidence = E ? E->AlignmentConfidence : 0.0f;
        const float Required = Au23RequiredAnchorConfidenceForClass(Class);
        const bool bStrongForClass = bHasEvidence && Confidence >= Required;
        const bool bCurrentHardAnchor = IsAu23CurrentHardAnchorClass(Class);
        const bool bProposedNewHardAnchor = bStrongForClass && !bCurrentHardAnchor &&
            (Class == "ROUNDED_VOWEL" || Class == "SH_CH_J" || Class == "FRONT_VOWEL");

        float PrevGapMs = -1.0f;
        float NextGapMs = -1.0f;
        for (int32 J = I - 1; J >= 0; --J)
        {
            if (StreamingTrack.Events[J].TextIslandIndex == S.TextIslandIndex)
            {
                PrevGapMs = (S.FinalRenderCenterSeconds - StreamingTrack.Events[J].FinalRenderCenterSeconds) * 1000.0f;
                break;
            }
        }
        for (int32 J = I + 1; J < StreamingTrack.Events.Num(); ++J)
        {
            if (StreamingTrack.Events[J].TextIslandIndex == S.TextIslandIndex)
            {
                NextGapMs = (StreamingTrack.Events[J].FinalRenderCenterSeconds - S.FinalRenderCenterSeconds) * 1000.0f;
                break;
            }
        }

        std::string Verdict = "no_evidence";
        if (bHasEvidence && bStrongForClass && bCurrentHardAnchor) { Verdict = "validated_current_hard_anchor"; }
        else if (bHasEvidence && bStrongForClass && bProposedNewHardAnchor) { Verdict = "candidate_new_hard_anchor"; }
        else if (bHasEvidence && bStrongForClass) { Verdict = "strong_diagnostic_only"; }
        else if (bHasEvidence) { Verdict = "weak_or_unstable_evidence"; }

        Out << S.EventIndex << ','
            << S.TextIslandIndex << ','
            << S.AudioIslandIndex << ','
            << CsvEscape(S.PoseID.ToString().ToStdString()) << ','
            << CsvEscape(S.SourceWord.ToStdString()) << ','
            << S.WordIndex << ','
            << S.PhraseIndex << ','
            << CsvEscape(Class) << ','
            << S.FinalRenderCenterSeconds << ','
            << EvidenceCenterSec << ','
            << TargetVisualCenterSec << ','
            << ErrorMs << ','
            << FMath::Abs(ErrorMs) << ','
            << RawShiftMs << ','
            << Confidence << ','
            << Required << ','
            << (bHasEvidence ? 1 : 0) << ','
            << (bStrongForClass ? 1 : 0) << ','
            << (bCurrentHardAnchor ? 1 : 0) << ','
            << (bProposedNewHardAnchor ? 1 : 0) << ','
            << CsvEscape(Verdict) << ','
            << PrevGapMs << ','
            << NextGapMs << "\n";
    }
}


static void WriteAlpha32NudgeEffectivenessCsv(
    const fs::path& Path,
    const FOffgridAIAlignedVisemeTrack& PreNudgeTrack,
    const FSubmittedPoseDiagnostics& PreNudgePoseDiagnostics,
    const FOffgridAIAlignedVisemeTrack& PostNudgeTrack,
    const FSubmittedPoseDiagnostics& PostNudgePoseDiagnostics)
{
    std::map<int32, const FOffgridAIAlignedVisemeEvent*> PostByEventIndex;
    for (const FOffgridAIAlignedVisemeEvent& E : PostNudgeTrack.Events)
    {
        PostByEventIndex[E.EventIndex] = &E;
    }

    std::ofstream Out(Path);
    Out << "EventIndex,PoseID,SourceWord,WordIndex,PhraseIndex,TextIslandIndex,AudioIslandIndex,LandmarkClass,PreCommittedCenterSec,PostCommittedCenterSec,PreFacePeakSec,PostFacePeakSec,TargetVisualCenterSec,OfflineEvidenceConfidence,NudgeRequiredConfidence,NudgeAccepted,NudgeReason,AppliedShiftMs,RawShiftMs,PreCommittedToTargetErrorMs,PostCommittedToTargetErrorMs,PreFaceToTargetErrorMs,PostFaceToTargetErrorMs,CommittedTargetImprovementMs,FaceTargetImprovementMs,PreFaceMinusCommittedMs,PostFaceMinusCommittedMs\n";

    for (const FOffgridAIAlignedVisemeEvent& Pre : PreNudgeTrack.Events)
    {
        if (!IsAlpha31PrimaryAuditPose(Pre.PoseID))
        {
            continue;
        }
        const auto PostIt = PostByEventIndex.find(Pre.EventIndex);
        const FOffgridAIAlignedVisemeEvent* Post = PostIt != PostByEventIndex.end() ? PostIt->second : nullptr;
        if (!Post)
        {
            continue;
        }

        const float TargetSec = Post->AudioNudgeCandidateRawCenterSeconds;
        const float Confidence = Post->AudioNudgeCandidateConfidence;
        const float RequiredConfidence = Post->AudioNudgeRequiredConfidence;
        const bool bStrongEvidence = TargetSec > 0.0f && Confidence >= RequiredConfidence && RequiredConfidence > 0.0f;
        if (!bStrongEvidence)
        {
            continue;
        }

        const auto PrePeakIt = PreNudgePoseDiagnostics.Peaks.find(Pre.EventIndex);
        const FPeakInfo* PrePeak = PrePeakIt != PreNudgePoseDiagnostics.Peaks.end() ? &PrePeakIt->second : nullptr;
        const bool bHasPrePeak = PrePeak && PrePeak->PeakWeight >= 0.0f;
        const float PreFaceSec = bHasPrePeak ? PeakWindowWeightedCenterTime(*PrePeak) : -1.0f;

        const auto PostPeakIt = PostNudgePoseDiagnostics.Peaks.find(Post->EventIndex);
        const FPeakInfo* PostPeak = PostPeakIt != PostNudgePoseDiagnostics.Peaks.end() ? &PostPeakIt->second : nullptr;
        const bool bHasPostPeak = PostPeak && PostPeak->PeakWeight >= 0.0f;
        const float PostFaceSec = bHasPostPeak ? PeakWindowWeightedCenterTime(*PostPeak) : -1.0f;

        const float PreCommittedErrMs = (Pre.FinalRenderCenterSeconds - TargetSec) * 1000.0f;
        const float PostCommittedErrMs = (Post->FinalRenderCenterSeconds - TargetSec) * 1000.0f;
        const float PreFaceErrMs = bHasPrePeak ? (PreFaceSec - TargetSec) * 1000.0f : 0.0f;
        const float PostFaceErrMs = bHasPostPeak ? (PostFaceSec - TargetSec) * 1000.0f : 0.0f;
        const float CommittedImprovementMs = FMath::Abs(PreCommittedErrMs) - FMath::Abs(PostCommittedErrMs);
        const float FaceImprovementMs = (bHasPrePeak && bHasPostPeak) ? (FMath::Abs(PreFaceErrMs) - FMath::Abs(PostFaceErrMs)) : 0.0f;

        Out << Pre.EventIndex << ','
            << CsvEscape(Post->PoseID.ToString().ToStdString()) << ','
            << CsvEscape(Post->SourceWord.ToStdString()) << ','
            << Post->WordIndex << ','
            << Post->PhraseIndex << ','
            << Post->TextIslandIndex << ','
            << Post->AudioIslandIndex << ','
            << CsvEscape(LandmarkClassForPose(Post->PoseID)) << ','
            << Pre.FinalRenderCenterSeconds << ','
            << Post->FinalRenderCenterSeconds << ','
            << PreFaceSec << ','
            << PostFaceSec << ','
            << TargetSec << ','
            << Confidence << ','
            << RequiredConfidence << ','
            << (WasNudgeAcceptedForEvent(*Post) ? 1 : 0) << ','
            << CsvEscape(NudgeReasonForEvent(*Post)) << ','
            << (Post->AudioNudgeCandidateAppliedShiftSeconds * 1000.0f) << ','
            << (Post->AudioNudgeCandidateRawShiftSeconds * 1000.0f) << ','
            << PreCommittedErrMs << ','
            << PostCommittedErrMs << ','
            << PreFaceErrMs << ','
            << PostFaceErrMs << ','
            << CommittedImprovementMs << ','
            << FaceImprovementMs << ','
            << (bHasPrePeak ? (PreFaceSec - Pre.FinalRenderCenterSeconds) * 1000.0f : 0.0f) << ','
            << (bHasPostPeak ? (PostFaceSec - Post->FinalRenderCenterSeconds) * 1000.0f : 0.0f) << "\n";
    }
}

static void WriteIslandLaunchDiagnosticsCsv(
    const fs::path& Path,
    const FOffgridAITextVisemePlan& Plan,
    const FOffgridAIAlignedVisemeTrack& Track,
    const TArray<FOffgridAIStreamingSpeechIsland>& SpeechIslands)
{
    struct FIslandSummary
    {
        int32 TextIslandIndex = INDEX_NONE;
        int32 AudioIslandIndex = INDEX_NONE;
        int32 FirstEventIndex = INDEX_NONE;
        int32 LastEventIndex = INDEX_NONE;
        int32 EventCount = 0;
        float FirstCenterSec = 0.0f;
        float LastCenterSec = 0.0f;
        float IslandAudioStartSec = 0.0f;
        float IslandAudioEndSec = 0.0f;
        float IslandTextStartSec = 0.0f;
        float IslandTextEndSec = 0.0f;
        float PlannerIslandPredictedDurationSec = 0.0f;
        float PlannerIslandPunctuationSec = 0.0f;
        float FirstEventLocalOffsetSec = 0.0f;
        float LastEventLocalOffsetSec = 0.0f;
    };

    std::map<int32, FIslandSummary> ByTextIsland;
    for (const FOffgridAIAlignedVisemeEvent& E : Track.Events)
    {
        FIslandSummary& S = ByTextIsland[E.TextIslandIndex];
        if (S.EventCount == 0)
        {
            S.TextIslandIndex = E.TextIslandIndex;
            S.AudioIslandIndex = E.AudioIslandIndex;
            S.FirstEventIndex = E.EventIndex;
            S.LastEventIndex = E.EventIndex;
            S.FirstCenterSec = E.FinalRenderCenterSeconds;
            S.IslandAudioStartSec = E.IslandAudioStartSeconds;
            S.IslandAudioEndSec = E.IslandAudioEndSeconds;
            S.IslandTextStartSec = E.IslandTextStartSeconds;
            S.IslandTextEndSec = E.IslandTextEndSeconds;
            S.PlannerIslandPredictedDurationSec = E.PlannerIslandPredictedDurationSeconds;
            S.PlannerIslandPunctuationSec = E.PlannerIslandPunctuationSeconds;
            S.FirstEventLocalOffsetSec = E.FinalRenderCenterSeconds - E.IslandAudioStartSeconds;
        }
        S.LastEventIndex = E.EventIndex;
        S.LastCenterSec = E.FinalRenderCenterSeconds;
        S.LastEventLocalOffsetSec = E.FinalRenderCenterSeconds - E.IslandAudioStartSeconds;
        S.EventCount += 1;
    }

    std::ofstream Out(Path);
    Out << "TextIslandIndex,AudioIslandIndex,FirstEventIndex,LastEventIndex,EventCount,LaunchSec,AudioIslandStartSec,AudioIslandEndSec,AudioIslandLastSpeechSec,AudioIslandDetectedSpanSec,LaunchDelayMs,FirstCenterSec,LastCenterSec,FirstEventLocalOffsetSec,LastEventLocalOffsetSec,IslandAudioEndSec,PredictedIslandDurationSec,PlannerPunctuationSec,AnimTailVsDetectorEndMs,IslandTextStartSec,IslandTextEndSec,LaunchReason\n";
    for (const auto& Pair : ByTextIsland)
    {
        const FIslandSummary& S = Pair.second;
        float AudioStartSec = 0.0f;
        float AudioEndSec = 0.0f;
        float AudioLastSpeechSec = 0.0f;
        bool bHasAudioStart = false;
        if (S.AudioIslandIndex >= 0)
        {
            for (const FOffgridAIStreamingSpeechIsland& Island : SpeechIslands)
            {
                if (Island.IslandIndex == S.AudioIslandIndex)
                {
                    AudioStartSec = Island.AudioBufferStartSec;
                    AudioEndSec = Island.AudioBufferEndSec;
                    AudioLastSpeechSec = Island.AudioBufferLastSpeechSec;
                    bHasAudioStart = true;
                    break;
                }
            }
        }
        const float DetectedSpanSec = bHasAudioStart ? FMath::Max(0.0f, AudioEndSec - AudioStartSec) : 0.0f;
        const float LaunchDelayMs = bHasAudioStart ? (S.IslandAudioStartSec - AudioStartSec) * 1000.0f : 0.0f;
        const float AnimTailVsDetectorEndMs = bHasAudioStart ? (S.LastCenterSec - AudioEndSec) * 1000.0f : 0.0f;
        const std::string Reason = bHasAudioStart ? "detected_audio_start" : "fallback_or_unmapped";
        Out << S.TextIslandIndex << ','
            << S.AudioIslandIndex << ','
            << S.FirstEventIndex << ','
            << S.LastEventIndex << ','
            << S.EventCount << ','
            << S.IslandAudioStartSec << ','
            << (bHasAudioStart ? AudioStartSec : 0.0f) << ','
            << (bHasAudioStart ? AudioEndSec : 0.0f) << ','
            << (bHasAudioStart ? AudioLastSpeechSec : 0.0f) << ','
            << DetectedSpanSec << ','
            << LaunchDelayMs << ','
            << S.FirstCenterSec << ','
            << S.LastCenterSec << ','
            << S.FirstEventLocalOffsetSec << ','
            << S.LastEventLocalOffsetSec << ','
            << S.IslandAudioEndSec << ','
            << S.PlannerIslandPredictedDurationSec << ','
            << S.PlannerIslandPunctuationSec << ','
            << AnimTailVsDetectorEndMs << ','
            << S.IslandTextStartSec << ','
            << S.IslandTextEndSec << ','
            << CsvEscape(Reason) << "\n";
    }
}



static void WritePhraseDurationDiagnosticsCsv(
    const fs::path& Path,
    const std::string& Transcript,
    const FOffgridAIAlignedVisemeTrack& Track,
    const FTimingDiagnostics& TimingDiagnostics)
{
    struct FEvidenceEventSummary
    {
        float StartSec = 0.0f;
        float CenterSec = 0.0f;
        float EndSec = 0.0f;
        bool bValid = false;
    };

    struct FPhraseSummary
    {
        int32 TextIslandIndex = INDEX_NONE;
        int32 PhraseIndex = INDEX_NONE;
        int32 FirstEventIndex = INDEX_NONE;
        int32 LastEventIndex = INDEX_NONE;
        int32 EventCount = 0;
        int32 FirstWordIndex = INDEX_NONE;
        int32 LastWordIndex = INDEX_NONE;
        FString FirstWord;
        FString LastWord;

        float PlannerPhraseSpeechDurationSec = 0.0f;
        float PlannerFirstOffsetSec = 0.0f;
        float PlannerLastOffsetSec = 0.0f;
        float PlannerIslandDurationSec = 0.0f;
        float PlannerIslandPunctuationSec = 0.0f;

        float CommittedFirstCenterSec = 0.0f;
        float CommittedLastCenterSec = 0.0f;
        float CommittedFirstRenderStartSec = 0.0f;
        float CommittedLastRenderEndSec = 0.0f;

        int32 EvidenceEventCount = 0;
        float EvidenceFirstCenterSec = 0.0f;
        float EvidenceLastCenterSec = 0.0f;
        float EvidenceFirstStartSec = 0.0f;
        float EvidenceLastEndSec = 0.0f;
    };

    std::map<int32, FEvidenceEventSummary> EvidenceByEventIndex;
    for (const FTimingDiagnosticRow& Row : TimingDiagnostics.Rows)
    {
        FEvidenceEventSummary Evidence;
        Evidence.StartSec = Row.EvidenceStartSec;
        Evidence.CenterSec = Row.EvidenceCenterSec;
        Evidence.EndSec = Row.EvidenceEndSec;
        Evidence.bValid = true;
        EvidenceByEventIndex[Row.EventIndex] = Evidence;
    }

    std::map<std::pair<int32, int32>, FPhraseSummary> ByPhrase;
    for (const FOffgridAIAlignedVisemeEvent& E : Track.Events)
    {
        FPhraseSummary& S = ByPhrase[std::make_pair(E.TextIslandIndex, E.PhraseIndex)];
        const bool bFirst = S.EventCount == 0;
        if (bFirst)
        {
            S.TextIslandIndex = E.TextIslandIndex;
            S.PhraseIndex = E.PhraseIndex;
            S.FirstEventIndex = E.EventIndex;
            S.FirstWordIndex = E.WordIndex;
            S.FirstWord = E.SourceWord;
            S.PlannerPhraseSpeechDurationSec = E.PlannerProsodyGroupAllocatedSeconds;
            S.PlannerFirstOffsetSec = E.PlannerEventPredictedOffsetSeconds;
            S.PlannerLastOffsetSec = E.PlannerEventPredictedOffsetSeconds;
            S.PlannerIslandDurationSec = E.PlannerIslandPredictedDurationSeconds;
            S.PlannerIslandPunctuationSec = E.PlannerIslandPunctuationSeconds;
            S.CommittedFirstCenterSec = E.FinalRenderCenterSeconds;
            S.CommittedLastCenterSec = E.FinalRenderCenterSeconds;
            S.CommittedFirstRenderStartSec = E.RenderStartSeconds;
            S.CommittedLastRenderEndSec = E.RenderEndSeconds;
        }

        S.LastEventIndex = E.EventIndex;
        S.LastWordIndex = E.WordIndex;
        S.LastWord = E.SourceWord;
        S.PlannerLastOffsetSec = E.PlannerEventPredictedOffsetSeconds;
        S.CommittedLastCenterSec = E.FinalRenderCenterSeconds;
        S.CommittedLastRenderEndSec = E.RenderEndSeconds;
        ++S.EventCount;

        const auto EvidenceIt = EvidenceByEventIndex.find(E.EventIndex);
        if (EvidenceIt != EvidenceByEventIndex.end() && EvidenceIt->second.bValid)
        {
            const FEvidenceEventSummary& Evidence = EvidenceIt->second;
            if (S.EvidenceEventCount == 0)
            {
                S.EvidenceFirstCenterSec = Evidence.CenterSec;
                S.EvidenceFirstStartSec = Evidence.StartSec;
            }
            S.EvidenceLastCenterSec = Evidence.CenterSec;
            S.EvidenceLastEndSec = Evidence.EndSec;
            ++S.EvidenceEventCount;
        }
    }

    const std::map<int32, FBoundaryPunctuationDiagnostic> BoundaryByWordIndex = BuildWordBoundaryPunctuationDiagnostics(Transcript);

    std::ofstream Out(Path);
    Out << "TextIslandIndex,PhraseIndex,FirstEventIndex,LastEventIndex,EventCount,EvidenceEventCount,FirstWord,LastWord,FirstWordIndex,LastWordIndex,"
        << "PlannerPhraseSpeechDurationSec,PlannerPhraseEventSpanSec,PlannerIslandDurationSec,PlannerIslandPunctuationSec,"
        << "CommittedPhraseCenterSpanSec,CommittedPhraseRenderSpanSec,EvidencePhraseCenterSpanSec,EvidencePhraseRenderSpanSec,"
        << "PlannerPhraseSpeechToEvidenceCenterRatio,PlannerPhraseSpeechToEvidenceRenderRatio,CommittedCenterToEvidenceCenterRatio,"
        << "PrevPhrasePlannerGapSec,PrevPhraseCommittedCenterGapSec,PrevPhraseEvidenceCenterGapSec,PrevPhraseGapDeltaCommittedVsEvidenceMs,"
        << "BoundaryPunctuationKind,BoundaryPunctuationCharIndex,BoundaryIsHardSentenceBreak\n";

    bool bHasPrev = false;
    FPhraseSummary Prev;
    for (const auto& Pair : ByPhrase)
    {
        const FPhraseSummary& S = Pair.second;
        const float PlannerPhraseEventSpanSec = FMath::Max(0.0f, S.PlannerLastOffsetSec - S.PlannerFirstOffsetSec);
        const float CommittedCenterSpanSec = FMath::Max(0.0f, S.CommittedLastCenterSec - S.CommittedFirstCenterSec);
        const float CommittedRenderSpanSec = FMath::Max(0.0f, S.CommittedLastRenderEndSec - S.CommittedFirstRenderStartSec);
        const float EvidenceCenterSpanSec = S.EvidenceEventCount > 0 ? FMath::Max(0.0f, S.EvidenceLastCenterSec - S.EvidenceFirstCenterSec) : 0.0f;
        const float EvidenceRenderSpanSec = S.EvidenceEventCount > 0 ? FMath::Max(0.0f, S.EvidenceLastEndSec - S.EvidenceFirstStartSec) : 0.0f;
        const float PlannerToEvidenceCenterRatio = EvidenceCenterSpanSec > 1e-6f ? S.PlannerPhraseSpeechDurationSec / EvidenceCenterSpanSec : 0.0f;
        const float PlannerToEvidenceRenderRatio = EvidenceRenderSpanSec > 1e-6f ? S.PlannerPhraseSpeechDurationSec / EvidenceRenderSpanSec : 0.0f;
        const float CommittedToEvidenceCenterRatio = EvidenceCenterSpanSec > 1e-6f ? CommittedCenterSpanSec / EvidenceCenterSpanSec : 0.0f;

        float PrevPlannerGapSec = 0.0f;
        float PrevCommittedCenterGapSec = 0.0f;
        float PrevEvidenceCenterGapSec = 0.0f;
        float PrevGapDeltaCommittedVsEvidenceMs = 0.0f;
        FBoundaryPunctuationDiagnostic Boundary;
        if (bHasPrev && Prev.TextIslandIndex == S.TextIslandIndex)
        {
            PrevPlannerGapSec = S.PlannerFirstOffsetSec - Prev.PlannerLastOffsetSec;
            PrevCommittedCenterGapSec = S.CommittedFirstCenterSec - Prev.CommittedLastCenterSec;
            PrevEvidenceCenterGapSec = (S.EvidenceEventCount > 0 && Prev.EvidenceEventCount > 0) ? (S.EvidenceFirstCenterSec - Prev.EvidenceLastCenterSec) : 0.0f;
            PrevGapDeltaCommittedVsEvidenceMs = (PrevCommittedCenterGapSec - PrevEvidenceCenterGapSec) * 1000.0f;
            const auto BoundaryIt = BoundaryByWordIndex.find(S.FirstWordIndex);
            if (BoundaryIt != BoundaryByWordIndex.end())
            {
                Boundary = BoundaryIt->second;
            }
        }

        Out << S.TextIslandIndex << ','
            << S.PhraseIndex << ','
            << S.FirstEventIndex << ','
            << S.LastEventIndex << ','
            << S.EventCount << ','
            << S.EvidenceEventCount << ','
            << CsvEscape(S.FirstWord.ToStdString()) << ','
            << CsvEscape(S.LastWord.ToStdString()) << ','
            << S.FirstWordIndex << ','
            << S.LastWordIndex << ','
            << S.PlannerPhraseSpeechDurationSec << ','
            << PlannerPhraseEventSpanSec << ','
            << S.PlannerIslandDurationSec << ','
            << S.PlannerIslandPunctuationSec << ','
            << CommittedCenterSpanSec << ','
            << CommittedRenderSpanSec << ','
            << EvidenceCenterSpanSec << ','
            << EvidenceRenderSpanSec << ','
            << PlannerToEvidenceCenterRatio << ','
            << PlannerToEvidenceRenderRatio << ','
            << CommittedToEvidenceCenterRatio << ','
            << PrevPlannerGapSec << ','
            << PrevCommittedCenterGapSec << ','
            << PrevEvidenceCenterGapSec << ','
            << PrevGapDeltaCommittedVsEvidenceMs << ','
            << CsvEscape(Boundary.Kind) << ','
            << Boundary.CharIndex << ','
            << (Boundary.bHardSentenceBreak ? 1 : 0) << "\n";

        Prev = S;
        bHasPrev = true;
    }
}


static void WriteSoftPunctuationBoundaryDiagnosticsCsv(
    const fs::path& Path,
    const std::string& Transcript,
    const FOffgridAIAlignedVisemeTrack& Track,
    const TArray<FOffgridAIStreamingSpeechIsland>& Islands,
    const FTimingDiagnostics& TimingDiagnostics,
    float AudioDurationSec)
{
    (void)Transcript;
    (void)Track;
    (void)Islands;
    (void)TimingDiagnostics;
    (void)AudioDurationSec;
    std::ofstream Out(Path);
    Out << "Disabled,Reason\n1,M09_audio_occupancy_runtime_has_no_soft_punctuation_boundary_model\n";
}

static void WriteRuntimeIslandAlignmentCsv(
    const fs::path& Path,
    const FOffgridAITextVisemePlan& Plan,
    const FOffgridAIAlignedVisemeTrack& Track,
    const TArray<FOffgridAIStreamingSpeechIsland>& SpeechIslands,
    const FTimingDiagnostics& TimingDiagnostics,
    float AudioDurationSec)
{
    struct FEvidenceEventSummary
    {
        float StartSec = 0.0f;
        float CenterSec = 0.0f;
        float EndSec = 0.0f;
        float StreamingErrorMs = 0.0f;
        bool bValid = false;
    };

    struct FRuntimeIslandAlignmentSummary
    {
        int32 TextIslandIndex = INDEX_NONE;
        int32 AudioIslandIndex = INDEX_NONE;
        int32 FirstEventIndex = INDEX_NONE;
        int32 LastEventIndex = INDEX_NONE;
        int32 EventCount = 0;
        int32 EvidenceEventCount = 0;
        FString FirstWord;
        FString LastWord;

        float PlannerDurationSec = 0.0f;
        float PlannerSpeechMaterialSec = 0.0f;
        float PlannerPunctuationSec = 0.0f;
        float PlannerShortUtteranceFloorSec = 0.0f;
        int32 PlannerSyllableCount = 0;
        float PlannerSyllableFloorSec = 0.0f;
        float PlannerFirstOffsetSec = 0.0f;
        float PlannerLastOffsetSec = 0.0f;

        float AudioStartSec = 0.0f;
        float AudioEndSec = 0.0f;
        float DetectorAudioStartSec = 0.0f;
        float DetectorAudioEndSec = 0.0f;
        bool bHasDetectorIsland = false;

        float CommittedFirstCenterSec = 0.0f;
        float CommittedLastCenterSec = 0.0f;
        float CommittedFirstRenderStartSec = 0.0f;
        float CommittedLastRenderEndSec = 0.0f;

        float EvidenceFirstCenterSec = 0.0f;
        float EvidenceLastCenterSec = 0.0f;
        float EvidenceFirstStartSec = 0.0f;
        float EvidenceLastEndSec = 0.0f;
        double StreamingAbsErrorSumMs = 0.0;
    };

    std::map<int32, FEvidenceEventSummary> EvidenceByEventIndex;
    for (const FTimingDiagnosticRow& Row : TimingDiagnostics.Rows)
    {
        FEvidenceEventSummary Evidence;
        Evidence.StartSec = Row.EvidenceStartSec;
        Evidence.CenterSec = Row.EvidenceCenterSec;
        Evidence.EndSec = Row.EvidenceEndSec;
        Evidence.StreamingErrorMs = Row.StreamingErrorMs;
        Evidence.bValid = true;
        EvidenceByEventIndex[Row.EventIndex] = Evidence;
    }

    std::map<int32, FRuntimeIslandAlignmentSummary> ByTextIsland;
    for (const FOffgridAIAlignedVisemeEvent& E : Track.Events)
    {
        FRuntimeIslandAlignmentSummary& S = ByTextIsland[E.TextIslandIndex];
        const bool bFirst = S.EventCount == 0;
        if (bFirst)
        {
            S.TextIslandIndex = E.TextIslandIndex;
            S.AudioIslandIndex = E.AudioIslandIndex;
            S.FirstEventIndex = E.EventIndex;
            S.LastEventIndex = E.EventIndex;
            S.FirstWord = E.SourceWord;
            S.LastWord = E.SourceWord;
            S.PlannerDurationSec = E.PlannerIslandPredictedDurationSeconds;
            S.PlannerSpeechMaterialSec = E.PlannerIslandSpeechMaterialSeconds;
            S.PlannerPunctuationSec = E.PlannerIslandPunctuationSeconds;
            S.PlannerShortUtteranceFloorSec = E.PlannerIslandShortUtteranceFloorSeconds;
            S.PlannerSyllableCount = E.PlannerIslandSyllableCount;
            S.PlannerSyllableFloorSec = E.PlannerIslandSyllableFloorSeconds;
            S.PlannerFirstOffsetSec = E.PlannerEventPredictedOffsetSeconds;
            S.PlannerLastOffsetSec = E.PlannerEventPredictedOffsetSeconds;
            S.AudioStartSec = E.IslandAudioStartSeconds;
            S.AudioEndSec = E.IslandAudioEndSeconds;
            S.CommittedFirstCenterSec = E.FinalRenderCenterSeconds;
            S.CommittedLastCenterSec = E.FinalRenderCenterSeconds;
            S.CommittedFirstRenderStartSec = E.RenderStartSeconds;
            S.CommittedLastRenderEndSec = E.RenderEndSeconds;
        }
        S.LastEventIndex = E.EventIndex;
        S.LastWord = E.SourceWord;
        S.AudioIslandIndex = E.AudioIslandIndex;
        S.AudioEndSec = E.IslandAudioEndSeconds;
        S.CommittedLastCenterSec = E.FinalRenderCenterSeconds;
        S.CommittedLastRenderEndSec = E.RenderEndSeconds;
        S.PlannerLastOffsetSec = E.PlannerEventPredictedOffsetSeconds;
        S.EventCount += 1;

        const auto EvidenceIt = EvidenceByEventIndex.find(E.EventIndex);
        if (EvidenceIt != EvidenceByEventIndex.end() && EvidenceIt->second.bValid)
        {
            const FEvidenceEventSummary& Evidence = EvidenceIt->second;
            if (S.EvidenceEventCount == 0)
            {
                S.EvidenceFirstCenterSec = Evidence.CenterSec;
                S.EvidenceFirstStartSec = Evidence.StartSec;
            }
            S.EvidenceLastCenterSec = Evidence.CenterSec;
            S.EvidenceLastEndSec = Evidence.EndSec;
            S.StreamingAbsErrorSumMs += FMath::Abs(Evidence.StreamingErrorMs);
            S.EvidenceEventCount += 1;
        }
    }

    for (auto& Pair : ByTextIsland)
    {
        FRuntimeIslandAlignmentSummary& S = Pair.second;
        if (S.AudioIslandIndex >= 0)
        {
            for (const FOffgridAIStreamingSpeechIsland& Island : SpeechIslands)
            {
                if (Island.IslandIndex == S.AudioIslandIndex)
                {
                    S.DetectorAudioStartSec = Island.AudioBufferStartSec;
                    S.DetectorAudioEndSec = GetSpeechIslandEndSec(Island, AudioDurationSec);
                    S.bHasDetectorIsland = true;
                    break;
                }
            }
        }
    }

    std::ofstream Out(Path);
    Out << "TextIslandIndex,AudioIslandIndex,FirstEventIndex,LastEventIndex,EventCount,EvidenceEventCount,FirstWord,LastWord,"
        << "PlannerDurationSec,PlannerSpeechMaterialSec,PlannerPunctuationSec,PlannerShortUtteranceFloorSec,PlannerSyllableCount,PlannerSyllableFloorSec,PlannerFirstOffsetSec,PlannerLastOffsetSec,PlannerEventSpanSec,"
        << "MappedAudioStartSec,MappedAudioEndSec,MappedAudioDurationSec,DetectorAudioStartSec,DetectorAudioEndSec,DetectorAudioDurationSec,"
        << "CommittedFirstCenterSec,CommittedLastCenterSec,CommittedCenterSpanSec,CommittedFirstRenderStartSec,CommittedLastRenderEndSec,CommittedRenderSpanSec,"
        << "EvidenceFirstCenterSec,EvidenceLastCenterSec,EvidenceCenterSpanSec,EvidenceFirstStartSec,EvidenceLastEndSec,EvidenceRenderSpanSec,"
        << "PlannerDurationToMappedAudioRatio,PlannerDurationToDetectorAudioRatio,CommittedSpanToEvidenceCenterSpanRatio,CommittedRenderSpanToEvidenceRenderSpanRatio,"
        << "FirstCenterErrorMs,LastCenterErrorMs,MeanStreamingAbsErrorMs,PrevMappedAudioGapSec,PrevDetectorAudioGapSec,PrevCommittedCenterGapSec,PrevEvidenceCenterGapSec,PrevGapDeltaCommittedVsEvidenceMs,PrevGapDeltaCommittedVsDetectorMs\n";

    bool bHasPrev = false;
    FRuntimeIslandAlignmentSummary Prev;
    for (const auto& Pair : ByTextIsland)
    {
        const FRuntimeIslandAlignmentSummary& S = Pair.second;
        const float PlannerEventSpanSec = FMath::Max(0.0f, S.PlannerLastOffsetSec - S.PlannerFirstOffsetSec);
        const float MappedAudioDurationSec = FMath::Max(0.0f, S.AudioEndSec - S.AudioStartSec);
        const float DetectorAudioDurationSec = S.bHasDetectorIsland ? FMath::Max(0.0f, S.DetectorAudioEndSec - S.DetectorAudioStartSec) : 0.0f;
        const float CommittedCenterSpanSec = FMath::Max(0.0f, S.CommittedLastCenterSec - S.CommittedFirstCenterSec);
        const float CommittedRenderSpanSec = FMath::Max(0.0f, S.CommittedLastRenderEndSec - S.CommittedFirstRenderStartSec);
        const float EvidenceCenterSpanSec = S.EvidenceEventCount > 0 ? FMath::Max(0.0f, S.EvidenceLastCenterSec - S.EvidenceFirstCenterSec) : 0.0f;
        const float EvidenceRenderSpanSec = S.EvidenceEventCount > 0 ? FMath::Max(0.0f, S.EvidenceLastEndSec - S.EvidenceFirstStartSec) : 0.0f;
        const float FirstCenterErrorMs = S.EvidenceEventCount > 0 ? (S.CommittedFirstCenterSec - S.EvidenceFirstCenterSec) * 1000.0f : 0.0f;
        const float LastCenterErrorMs = S.EvidenceEventCount > 0 ? (S.CommittedLastCenterSec - S.EvidenceLastCenterSec) * 1000.0f : 0.0f;
        const float MeanStreamingAbsErrorMs = S.EvidenceEventCount > 0 ? static_cast<float>(S.StreamingAbsErrorSumMs / static_cast<double>(S.EvidenceEventCount)) : 0.0f;
        const float PlannerToMappedRatio = MappedAudioDurationSec > 1e-6f ? S.PlannerDurationSec / MappedAudioDurationSec : 0.0f;
        const float PlannerToDetectorRatio = DetectorAudioDurationSec > 1e-6f ? S.PlannerDurationSec / DetectorAudioDurationSec : 0.0f;
        const float CommittedToEvidenceCenterRatio = EvidenceCenterSpanSec > 1e-6f ? CommittedCenterSpanSec / EvidenceCenterSpanSec : 0.0f;
        const float CommittedRenderToEvidenceRenderRatio = EvidenceRenderSpanSec > 1e-6f ? CommittedRenderSpanSec / EvidenceRenderSpanSec : 0.0f;

        float PrevMappedAudioGapSec = 0.0f;
        float PrevDetectorAudioGapSec = 0.0f;
        float PrevCommittedCenterGapSec = 0.0f;
        float PrevEvidenceCenterGapSec = 0.0f;
        float PrevGapDeltaCommittedVsEvidenceMs = 0.0f;
        float PrevGapDeltaCommittedVsDetectorMs = 0.0f;
        if (bHasPrev)
        {
            PrevMappedAudioGapSec = S.AudioStartSec - Prev.AudioEndSec;
            PrevDetectorAudioGapSec = (S.bHasDetectorIsland && Prev.bHasDetectorIsland) ? (S.DetectorAudioStartSec - Prev.DetectorAudioEndSec) : 0.0f;
            PrevCommittedCenterGapSec = S.CommittedFirstCenterSec - Prev.CommittedLastCenterSec;
            PrevEvidenceCenterGapSec = (S.EvidenceEventCount > 0 && Prev.EvidenceEventCount > 0) ? (S.EvidenceFirstCenterSec - Prev.EvidenceLastCenterSec) : 0.0f;
            PrevGapDeltaCommittedVsEvidenceMs = (PrevCommittedCenterGapSec - PrevEvidenceCenterGapSec) * 1000.0f;
            PrevGapDeltaCommittedVsDetectorMs = (PrevCommittedCenterGapSec - PrevDetectorAudioGapSec) * 1000.0f;
        }

        Out << S.TextIslandIndex << ','
            << S.AudioIslandIndex << ','
            << S.FirstEventIndex << ','
            << S.LastEventIndex << ','
            << S.EventCount << ','
            << S.EvidenceEventCount << ','
            << CsvEscape(S.FirstWord.ToStdString()) << ','
            << CsvEscape(S.LastWord.ToStdString()) << ','
            << S.PlannerDurationSec << ','
            << S.PlannerSpeechMaterialSec << ','
            << S.PlannerPunctuationSec << ','
            << S.PlannerShortUtteranceFloorSec << ','
            << S.PlannerSyllableCount << ','
            << S.PlannerSyllableFloorSec << ','
            << S.PlannerFirstOffsetSec << ','
            << S.PlannerLastOffsetSec << ','
            << PlannerEventSpanSec << ','
            << S.AudioStartSec << ','
            << S.AudioEndSec << ','
            << MappedAudioDurationSec << ','
            << S.DetectorAudioStartSec << ','
            << S.DetectorAudioEndSec << ','
            << DetectorAudioDurationSec << ','
            << S.CommittedFirstCenterSec << ','
            << S.CommittedLastCenterSec << ','
            << CommittedCenterSpanSec << ','
            << S.CommittedFirstRenderStartSec << ','
            << S.CommittedLastRenderEndSec << ','
            << CommittedRenderSpanSec << ','
            << S.EvidenceFirstCenterSec << ','
            << S.EvidenceLastCenterSec << ','
            << EvidenceCenterSpanSec << ','
            << S.EvidenceFirstStartSec << ','
            << S.EvidenceLastEndSec << ','
            << EvidenceRenderSpanSec << ','
            << PlannerToMappedRatio << ','
            << PlannerToDetectorRatio << ','
            << CommittedToEvidenceCenterRatio << ','
            << CommittedRenderToEvidenceRenderRatio << ','
            << FirstCenterErrorMs << ','
            << LastCenterErrorMs << ','
            << MeanStreamingAbsErrorMs << ','
            << PrevMappedAudioGapSec << ','
            << PrevDetectorAudioGapSec << ','
            << PrevCommittedCenterGapSec << ','
            << PrevEvidenceCenterGapSec << ','
            << PrevGapDeltaCommittedVsEvidenceMs << ','
            << PrevGapDeltaCommittedVsDetectorMs << "\n";

        Prev = S;
        bHasPrev = true;
    }
}


static float N10SpeechIslandStartSec(const FOffgridAIStreamingSpeechIsland& Island)
{
    return Island.AudioBufferStartSec;
}

static float N10SpeechIslandEndSec(const FOffgridAIStreamingSpeechIsland& Island, float ObservedAudioEndSec)
{
    return GetSpeechIslandEndSec(Island, ObservedAudioEndSec);
}

static void N10ObservedSpeechBounds(
    const TArray<FOffgridAIStreamingSpeechIsland>& Islands,
    float ObservedAudioEndSec,
    float& OutStartSec,
    float& OutEndSec)
{
    OutStartSec = 0.0f;
    OutEndSec = 0.0f;
    bool bAny = false;
    for (const FOffgridAIStreamingSpeechIsland& Island : Islands)
    {
        if (!Island.bStarted)
        {
            continue;
        }
        const float StartSec = N10SpeechIslandStartSec(Island);
        const float EndSec = N10SpeechIslandEndSec(Island, ObservedAudioEndSec);
        if (EndSec <= StartSec)
        {
            continue;
        }
        if (!bAny)
        {
            OutStartSec = StartSec;
            OutEndSec = EndSec;
            bAny = true;
        }
        else
        {
            OutStartSec = FMath::Min(OutStartSec, StartSec);
            OutEndSec = FMath::Max(OutEndSec, EndSec);
        }
    }
}

static float N10IntervalOverlapSec(float A0, float A1, float B0, float B1)
{
    return FMath::Max(0.0f, FMath::Min(A1, B1) - FMath::Max(A0, B0));
}

static float N10EventPauseOverlapSec(
    const FOffgridAIAlignedVisemeEvent& Event,
    const TArray<FOffgridAIStreamingSpeechIsland>& Islands,
    float ObservedAudioEndSec)
{
    const float EventStart = FMath::Max(0.0f, Event.RenderStartSeconds);
    const float EventEnd = FMath::Max(EventStart, Event.RenderEndSeconds);
    float SpeechOverlap = 0.0f;
    for (const FOffgridAIStreamingSpeechIsland& Island : Islands)
    {
        if (!Island.bStarted)
        {
            continue;
        }
        SpeechOverlap += N10IntervalOverlapSec(EventStart, EventEnd, N10SpeechIslandStartSec(Island), N10SpeechIslandEndSec(Island, ObservedAudioEndSec));
    }
    return FMath::Max(0.0f, EventEnd - EventStart - SpeechOverlap);
}

static float N10SpeechWithoutFreshVisemeSec(
    const TArray<FOffgridAIStreamingSpeechIsland>& Islands,
    const FOffgridAIAlignedVisemeTrack& Track,
    float ObservedAudioEndSec)
{
    constexpr float FreshRadiusSec = 0.120f;
    constexpr float StepSec = 0.010f;
    float MissingSec = 0.0f;
    for (const FOffgridAIStreamingSpeechIsland& Island : Islands)
    {
        if (!Island.bStarted)
        {
            continue;
        }
        const float StartSec = N10SpeechIslandStartSec(Island);
        const float EndSec = N10SpeechIslandEndSec(Island, ObservedAudioEndSec);
        for (float T = StartSec; T < EndSec; T += StepSec)
        {
            const float SampleT = T + StepSec * 0.5f;
            bool bFresh = false;
            for (const FOffgridAIAlignedVisemeEvent& Event : Track.Events)
            {
                if (FMath::Abs(Event.FinalRenderCenterSeconds - SampleT) <= FreshRadiusSec)
                {
                    bFresh = true;
                    break;
                }
            }
            if (!bFresh)
            {
                MissingSec += FMath::Min(StepSec, EndSec - T);
            }
        }
    }
    return MissingSec;
}

static void WriteN10TimingCoverageSummaryCsv(
    const fs::path& Path,
    const FOffgridAIAlignedVisemeTrack& Track,
    const TArray<FOffgridAIStreamingSpeechIsland>& Islands,
    float ObservedAudioEndSec)
{
    float ObservedStartSec = 0.0f;
    float ObservedEndSec = 0.0f;
    N10ObservedSpeechBounds(Islands, ObservedAudioEndSec, ObservedStartSec, ObservedEndSec);

    float FirstCenterSec = 0.0f;
    float LastCenterSec = 0.0f;
    bool bHasEvent = false;
    int TailFlushCount = 0;
    float PauseOverlapSec = 0.0f;
    for (const FOffgridAIAlignedVisemeEvent& Event : Track.Events)
    {
        if (!bHasEvent)
        {
            FirstCenterSec = Event.FinalRenderCenterSeconds;
            LastCenterSec = Event.FinalRenderCenterSeconds;
            bHasEvent = true;
        }
        else
        {
            FirstCenterSec = FMath::Min(FirstCenterSec, Event.FinalRenderCenterSeconds);
            LastCenterSec = FMath::Max(LastCenterSec, Event.FinalRenderCenterSeconds);
        }
        if (Event.CommitReason == FName(TEXT("end_of_stream_flush")) || Event.CommitReason.ToString().Contains(TEXT("tail_drain")))
        {
            ++TailFlushCount;
        }
        PauseOverlapSec += N10EventPauseOverlapSec(Event, Islands, ObservedAudioEndSec);
    }

    const float StartsBeforeMs = bHasEvent ? FMath::Max(0.0f, ObservedStartSec - FirstCenterSec) * 1000.0f : 0.0f;
    const float StartsAfterMs = bHasEvent ? FMath::Max(0.0f, FirstCenterSec - ObservedStartSec) * 1000.0f : 0.0f;
    const float TailUncoveredMs = bHasEvent ? FMath::Max(0.0f, ObservedEndSec - LastCenterSec) * 1000.0f : 0.0f;
    const float PastEndMs = bHasEvent ? FMath::Max(0.0f, LastCenterSec - ObservedEndSec) * 1000.0f : 0.0f;
    const float SpeechGapMs = N10SpeechWithoutFreshVisemeSec(Islands, Track, ObservedAudioEndSec) * 1000.0f;

    std::ofstream Out(Path);
    Out << "LineID,ObservedAudioStartSec,ObservedAudioEndSec,FirstCommittedCenterSec,LastCommittedCenterSec,AnimationStartsBeforeSpeechMs,AnimationStartsAfterSpeechMs,AudioTailUncoveredMs,AnimationPastAudioEndMs,SpeechActiveWithoutFreshVisemeMs,AnimationDuringPauseMs,TailFlushCount\n";
    Out << CsvEscape(Track.LineID.ToString().ToStdString()) << ','
        << ObservedStartSec << ','
        << ObservedEndSec << ','
        << FirstCenterSec << ','
        << LastCenterSec << ','
        << StartsBeforeMs << ','
        << StartsAfterMs << ','
        << TailUncoveredMs << ','
        << PastEndMs << ','
        << SpeechGapMs << ','
        << (PauseOverlapSec * 1000.0f) << ','
        << TailFlushCount << "\n";
}

static void WriteN10TimingPauseOverlapCsv(
    const fs::path& Path,
    const FOffgridAIAlignedVisemeTrack& Track,
    const TArray<FOffgridAIStreamingSpeechIsland>& Islands,
    float ObservedAudioEndSec)
{
    std::ofstream Out(Path);
    Out << "LineID,EventIndex,PoseID,SourceWord,CenterSec,RenderStartSec,RenderEndSec,InsideDetectedPause,DistanceToNearestSpeechMs,PauseOverlapMs,PauseOverlapSeverity\n";
    for (const FOffgridAIAlignedVisemeEvent& Event : Track.Events)
    {
        const bool bInsideSpeech = IsCenterInsideSpeechIsland(Islands, Event.FinalRenderCenterSeconds, ObservedAudioEndSec, 0.0f, 0.0f);
        const float DistanceMs = bInsideSpeech ? 0.0f : DistanceToNearestSpeechIslandMs(Islands, Event.FinalRenderCenterSeconds, ObservedAudioEndSec);
        const float PauseOverlapMs = N10EventPauseOverlapSec(Event, Islands, ObservedAudioEndSec) * 1000.0f;
        FString Severity = TEXT("none");
        if (!bInsideSpeech && DistanceMs > 120.0f)
        {
            Severity = TEXT("high");
        }
        else if (!bInsideSpeech && DistanceMs > 60.0f)
        {
            Severity = TEXT("medium");
        }
        else if (!bInsideSpeech || PauseOverlapMs > 30.0f)
        {
            Severity = TEXT("low");
        }
        Out << CsvEscape(Track.LineID.ToString().ToStdString()) << ','
            << Event.EventIndex << ','
            << CsvEscape(Event.PoseID.ToString().ToStdString()) << ','
            << CsvEscape(Event.SourceWord.ToStdString()) << ','
            << Event.FinalRenderCenterSeconds << ','
            << Event.RenderStartSeconds << ','
            << Event.RenderEndSeconds << ','
            << (!bInsideSpeech ? 1 : 0) << ','
            << DistanceMs << ','
            << PauseOverlapMs << ','
            << CsvEscape(Severity.ToStdString()) << "\n";
    }
}

static void WriteN10TimingSpeechGapsCsv(
    const fs::path& Path,
    const FOffgridAIAlignedVisemeTrack& Track,
    const TArray<FOffgridAIStreamingSpeechIsland>& Islands,
    float ObservedAudioEndSec)
{
    constexpr float FreshRadiusSec = 0.120f;
    constexpr float MinGapSec = 0.080f;
    std::ofstream Out(Path);
    Out << "LineID,SpeechGapStartSec,SpeechGapEndSec,GapDurationMs,SpeechIslandIndex,PreviousViseme,NextViseme\n";
    for (int32 IslandI = 0; IslandI < Islands.Num(); ++IslandI)
    {
        const FOffgridAIStreamingSpeechIsland& Island = Islands[IslandI];
        if (!Island.bStarted)
        {
            continue;
        }
        const float SpeechStart = N10SpeechIslandStartSec(Island);
        const float SpeechEnd = N10SpeechIslandEndSec(Island, ObservedAudioEndSec);
        float Cursor = SpeechStart;
        for (const FOffgridAIAlignedVisemeEvent& Event : Track.Events)
        {
            const float CoverStart = FMath::Max(SpeechStart, Event.FinalRenderCenterSeconds - FreshRadiusSec);
            const float CoverEnd = FMath::Min(SpeechEnd, Event.FinalRenderCenterSeconds + FreshRadiusSec);
            if (CoverEnd <= SpeechStart || CoverStart >= SpeechEnd)
            {
                continue;
            }
            if (CoverStart - Cursor >= MinGapSec)
            {
                const FOffgridAIAlignedVisemeEvent* Prev = nullptr;
                const FOffgridAIAlignedVisemeEvent* Next = nullptr;
                for (const FOffgridAIAlignedVisemeEvent& Candidate : Track.Events)
                {
                    if (Candidate.FinalRenderCenterSeconds <= Cursor)
                    {
                        Prev = &Candidate;
                    }
                    if (!Next && Candidate.FinalRenderCenterSeconds >= CoverStart)
                    {
                        Next = &Candidate;
                    }
                }
                Out << CsvEscape(Track.LineID.ToString().ToStdString()) << ','
                    << Cursor << ','
                    << CoverStart << ','
                    << ((CoverStart - Cursor) * 1000.0f) << ','
                    << IslandI << ','
                    << CsvEscape(Prev ? Prev->PoseID.ToString().ToStdString() : std::string()) << ','
                    << CsvEscape(Next ? Next->PoseID.ToString().ToStdString() : std::string()) << "\n";
            }
            Cursor = FMath::Max(Cursor, CoverEnd);
        }
        if (SpeechEnd - Cursor >= MinGapSec)
        {
            const FOffgridAIAlignedVisemeEvent* Prev = nullptr;
            for (const FOffgridAIAlignedVisemeEvent& Candidate : Track.Events)
            {
                if (Candidate.FinalRenderCenterSeconds <= Cursor)
                {
                    Prev = &Candidate;
                }
            }
            Out << CsvEscape(Track.LineID.ToString().ToStdString()) << ','
                << Cursor << ','
                << SpeechEnd << ','
                << ((SpeechEnd - Cursor) * 1000.0f) << ','
                << IslandI << ','
                << CsvEscape(Prev ? Prev->PoseID.ToString().ToStdString() : std::string()) << ','
                << "\n";
        }
    }
}

static void N10WriteRegionRow(
    std::ofstream& Out,
    const FOffgridAIAlignedVisemeTrack& Track,
    const TArray<FOffgridAIStreamingSpeechIsland>& Islands,
    float ObservedAudioEndSec,
    const char* RegionType,
    int32 RegionIndex,
    int32 FirstEventInclusive,
    int32 LastEventInclusive)
{
    if (FirstEventInclusive < 0 || LastEventInclusive < FirstEventInclusive || Track.Events.Num() <= 0)
    {
        return;
    }
    float TextStart = std::numeric_limits<float>::max();
    float TextEnd = 0.0f;
    float FirstViseme = std::numeric_limits<float>::max();
    float LastViseme = 0.0f;
    float AudioStart = std::numeric_limits<float>::max();
    float AudioEnd = 0.0f;
    bool bHasAudio = false;
    int Count = 0;
    for (const FOffgridAIAlignedVisemeEvent& Event : Track.Events)
    {
        if (Event.EventIndex < FirstEventInclusive || Event.EventIndex > LastEventInclusive)
        {
            continue;
        }
        ++Count;
        TextStart = FMath::Min(TextStart, Event.TextDiagnosticCenterSeconds);
        TextEnd = FMath::Max(TextEnd, Event.TextDiagnosticCenterSeconds);
        FirstViseme = FMath::Min(FirstViseme, Event.FinalRenderCenterSeconds);
        LastViseme = FMath::Max(LastViseme, Event.FinalRenderCenterSeconds);
        if (Islands.IsValidIndex(Event.AudioIslandIndex))
        {
            bHasAudio = true;
            AudioStart = FMath::Min(AudioStart, N10SpeechIslandStartSec(Islands[Event.AudioIslandIndex]));
            AudioEnd = FMath::Max(AudioEnd, N10SpeechIslandEndSec(Islands[Event.AudioIslandIndex], ObservedAudioEndSec));
        }
    }
    if (Count <= 0)
    {
        return;
    }
    if (!bHasAudio)
    {
        AudioStart = FirstViseme;
        AudioEnd = LastViseme;
    }
    const float LeadingGapMs = (FirstViseme - AudioStart) * 1000.0f;
    const float TrailingGapMs = (AudioEnd - LastViseme) * 1000.0f;
    const float CoverageRatio = (AudioEnd > AudioStart) ? FMath::Clamp((LastViseme - FirstViseme) / (AudioEnd - AudioStart), 0.0f, 10.0f) : 0.0f;
    Out << CsvEscape(Track.LineID.ToString().ToStdString()) << ','
        << CsvEscape(std::string(RegionType)) << ','
        << RegionIndex << ','
        << TextStart << ','
        << TextEnd << ','
        << AudioStart << ','
        << AudioEnd << ','
        << FirstViseme << ','
        << LastViseme << ','
        << LeadingGapMs << ','
        << TrailingGapMs << ','
        << CoverageRatio << ','
        << Count << "\n";
}

static void WriteN10TimingCoverageRegionsCsv(
    const fs::path& Path,
    const FOffgridAIAlignedVisemeTrack& Track,
    const TArray<FOffgridAIStreamingSpeechIsland>& Islands,
    float ObservedAudioEndSec)
{
    std::ofstream Out(Path);
    Out << "LineID,RegionType,RegionIndex,TextRegionStartSec,TextRegionEndSec,AudioRegionStartSec,AudioRegionEndSec,FirstVisemeCenterSec,LastVisemeCenterSec,LeadingGapMs,TrailingGapMs,CoverageRatio,EventCount\n";

    std::map<int32, std::pair<int32, int32>> TextIslandRanges;
    for (const FOffgridAIAlignedVisemeEvent& Event : Track.Events)
    {
        const int32 Key = Event.TextIslandIndex != INDEX_NONE ? Event.TextIslandIndex : 0;
        auto& Range = TextIslandRanges[Key];
        if (Range.second == 0 && Range.first == 0 && Event.EventIndex != 0)
        {
            Range.first = Event.EventIndex;
            Range.second = Event.EventIndex;
        }
        else
        {
            Range.first = FMath::Min(Range.first, Event.EventIndex);
            Range.second = FMath::Max(Range.second, Event.EventIndex);
        }
    }
    for (const auto& It : TextIslandRanges)
    {
        N10WriteRegionRow(Out, Track, Islands, ObservedAudioEndSec, "punctuation", It.first, It.second.first, It.second.second);
    }

    constexpr int32 SyntheticRegionEventCount = 8;
    int32 RegionIndex = 0;
    for (int32 Start = 0; Start < Track.Events.Num(); Start += SyntheticRegionEventCount)
    {
        const int32 FirstEvent = Track.Events[Start].EventIndex;
        const int32 LastEvent = Track.Events[FMath::Min(Start + SyntheticRegionEventCount - 1, Track.Events.Num() - 1)].EventIndex;
        N10WriteRegionRow(Out, Track, Islands, ObservedAudioEndSec, "synthetic", RegionIndex++, FirstEvent, LastEvent);
    }
}

static void WriteN10TimingCoverageDiagnostics(
    const fs::path& CaseOutDir,
    const FOffgridAIAlignedVisemeTrack& Track,
    const TArray<FOffgridAIStreamingSpeechIsland>& Islands,
    float ObservedAudioEndSec)
{
    WriteN10TimingCoverageSummaryCsv(CaseOutDir / "timing_coverage_summary.csv", Track, Islands, ObservedAudioEndSec);
    WriteN10TimingCoverageRegionsCsv(CaseOutDir / "timing_coverage_regions.csv", Track, Islands, ObservedAudioEndSec);
    WriteN10TimingPauseOverlapCsv(CaseOutDir / "timing_pause_overlap.csv", Track, Islands, ObservedAudioEndSec);
    WriteN10TimingSpeechGapsCsv(CaseOutDir / "timing_speech_gaps.csv", Track, Islands, ObservedAudioEndSec);
}


static float N11PlanCenterSec(const FOffgridAITextVisemePlan& Plan, int32 EventIndex)
{
    if (!Plan.Events.IsValidIndex(EventIndex))
    {
        return 0.0f;
    }
    const FOffgridAITextVisemeEvent& E = Plan.Events[EventIndex];
    return ((FMath::Clamp(E.StartNorm, 0.0f, 1.0f) + FMath::Clamp(E.EndNorm, E.StartNorm, 1.0f)) * 0.5f) * FMath::Max(Plan.EstimatedDurationSeconds, 0.001f);
}

static float N11PlanStartSec(const FOffgridAITextVisemePlan& Plan, int32 EventIndex)
{
    if (!Plan.Events.IsValidIndex(EventIndex))
    {
        return 0.0f;
    }
    return FMath::Clamp(Plan.Events[EventIndex].StartNorm, 0.0f, 1.0f) * FMath::Max(Plan.EstimatedDurationSeconds, 0.001f);
}

static float N11PlanEndSec(const FOffgridAITextVisemePlan& Plan, int32 EventIndex)
{
    if (!Plan.Events.IsValidIndex(EventIndex))
    {
        return 0.0f;
    }
    const FOffgridAITextVisemeEvent& E = Plan.Events[EventIndex];
    return FMath::Clamp(E.EndNorm, E.StartNorm, 1.0f) * FMath::Max(Plan.EstimatedDurationSeconds, 0.001f);
}

static float N11ObservedActiveSec(const TArray<FOffgridAIStreamingSpeechIsland>& Islands, float ObservedAudioEndSec)
{
    float Sum = 0.0f;
    for (const FOffgridAIStreamingSpeechIsland& Island : Islands)
    {
        if (!Island.bStarted)
        {
            continue;
        }
        const float Start = N10SpeechIslandStartSec(Island);
        const float End = N10SpeechIslandEndSec(Island, ObservedAudioEndSec);
        Sum += FMath::Max(0.0f, End - Start);
    }
    return Sum;
}

static void WriteN11OccupancyPacingMetricsCsv(
    const fs::path& Path,
    const FOffgridAITextVisemePlan& Plan,
    const FOffgridAIAlignedVisemeTrack& Track,
    const TArray<FOffgridAIStreamingSpeechIsland>& Islands,
    float ObservedAudioEndSec)
{
    std::ofstream Out(Path);
    Out << "LineID,SpeechIslandIndex,AudioStartSec,AudioEndSec,AudioDurationSec,PrevPauseSec,NextPauseSec,EventCount,FirstEventIndex,LastEventIndex,PlanStartSec,PlanEndSec,PlanDurationSec,CommittedStartSec,CommittedEndSec,CommittedDurationSec,ObservedToPlanScale,CommittedToAudioScale\n";
    for (int32 IslandI = 0; IslandI < Islands.Num(); ++IslandI)
    {
        const FOffgridAIStreamingSpeechIsland& Island = Islands[IslandI];
        if (!Island.bStarted)
        {
            continue;
        }
        const float AudioStart = N10SpeechIslandStartSec(Island);
        const float AudioEnd = N10SpeechIslandEndSec(Island, ObservedAudioEndSec);
        const float AudioDur = FMath::Max(0.0f, AudioEnd - AudioStart);
        float PrevPause = 0.0f;
        if (IslandI > 0 && Islands[IslandI - 1].bStarted)
        {
            PrevPause = FMath::Max(0.0f, AudioStart - N10SpeechIslandEndSec(Islands[IslandI - 1], ObservedAudioEndSec));
        }
        float NextPause = 0.0f;
        if (IslandI + 1 < Islands.Num() && Islands[IslandI + 1].bStarted)
        {
            NextPause = FMath::Max(0.0f, N10SpeechIslandStartSec(Islands[IslandI + 1]) - AudioEnd);
        }

        int32 Count = 0;
        int32 FirstEvent = INDEX_NONE;
        int32 LastEvent = INDEX_NONE;
        float PlanStart = std::numeric_limits<float>::max();
        float PlanEnd = 0.0f;
        float CommitStart = std::numeric_limits<float>::max();
        float CommitEnd = 0.0f;
        for (const FOffgridAIAlignedVisemeEvent& Event : Track.Events)
        {
            if (Event.AudioIslandIndex != IslandI)
            {
                continue;
            }
            ++Count;
            FirstEvent = FirstEvent == INDEX_NONE ? Event.EventIndex : FMath::Min(FirstEvent, Event.EventIndex);
            LastEvent = LastEvent == INDEX_NONE ? Event.EventIndex : FMath::Max(LastEvent, Event.EventIndex);
            PlanStart = FMath::Min(PlanStart, N11PlanStartSec(Plan, Event.EventIndex));
            PlanEnd = FMath::Max(PlanEnd, N11PlanEndSec(Plan, Event.EventIndex));
            CommitStart = FMath::Min(CommitStart, Event.FinalRenderCenterSeconds);
            CommitEnd = FMath::Max(CommitEnd, Event.FinalRenderCenterSeconds);
        }
        if (Count == 0)
        {
            PlanStart = PlanEnd = CommitStart = CommitEnd = 0.0f;
        }
        const float PlanDur = FMath::Max(0.0f, PlanEnd - PlanStart);
        const float CommitDur = FMath::Max(0.0f, CommitEnd - CommitStart);
        Out << CsvEscape(Track.LineID.ToString().ToStdString()) << ','
            << IslandI << ',' << AudioStart << ',' << AudioEnd << ',' << AudioDur << ',' << PrevPause << ',' << NextPause << ','
            << Count << ',' << FirstEvent << ',' << LastEvent << ','
            << PlanStart << ',' << PlanEnd << ',' << PlanDur << ','
            << CommitStart << ',' << CommitEnd << ',' << CommitDur << ','
            << (PlanDur > 0.001f ? AudioDur / PlanDur : 0.0f) << ','
            << (AudioDur > 0.001f ? CommitDur / AudioDur : 0.0f) << "\n";
    }
}

static void WriteN11RegionPacingOpportunitiesCsv(
    const fs::path& Path,
    const FOffgridAITextVisemePlan& Plan,
    const FOffgridAIAlignedVisemeTrack& Track,
    const TArray<FOffgridAIStreamingSpeechIsland>& Islands,
    const TArray<FOffgridAIAudioLandmarkCandidate>& StreamingLandmarks,
    float ObservedAudioEndSec,
    float PrerollSec)
{
    std::ofstream Out(Path);
    Out << "LineID,RegionType,RegionIndex,FirstEventIndex,LastEventIndex,EventCount,PlanStartSec,PlanEndSec,PlanDurationSec,AudioStartSec,AudioEndSec,AudioDurationSec,ObservedToPlanScale,CommittedStartSec,CommittedEndSec,CommittedDurationSec,LeadingGapMs,TrailingGapMs,LandmarkEventCount,MatchedLandmarkCount,UsableLead50Count,MeanAnchorDeltaMs\n";
    const auto AnchorRows = BuildAU37AnchorMatches(Plan, Track, StreamingLandmarks, PrerollSec);
    auto AppendRange = [&](const char* RegionType, int32 RegionIndex, int32 FirstArrayIndex, int32 LastArrayIndex)
    {
        if (!Track.Events.IsValidIndex(FirstArrayIndex) || !Track.Events.IsValidIndex(LastArrayIndex) || LastArrayIndex < FirstArrayIndex)
        {
            return;
        }
        const int32 FirstEvent = Track.Events[FirstArrayIndex].EventIndex;
        const int32 LastEvent = Track.Events[LastArrayIndex].EventIndex;
        float PlanStart = std::numeric_limits<float>::max();
        float PlanEnd = 0.0f;
        float CommitStart = std::numeric_limits<float>::max();
        float CommitEnd = 0.0f;
        float AudioStart = std::numeric_limits<float>::max();
        float AudioEnd = 0.0f;
        bool bHasAudio = false;
        std::set<int32> RegionEvents;
        for (int32 I = FirstArrayIndex; I <= LastArrayIndex; ++I)
        {
            const FOffgridAIAlignedVisemeEvent& Event = Track.Events[I];
            RegionEvents.insert(Event.EventIndex);
            PlanStart = FMath::Min(PlanStart, N11PlanStartSec(Plan, Event.EventIndex));
            PlanEnd = FMath::Max(PlanEnd, N11PlanEndSec(Plan, Event.EventIndex));
            CommitStart = FMath::Min(CommitStart, Event.FinalRenderCenterSeconds);
            CommitEnd = FMath::Max(CommitEnd, Event.FinalRenderCenterSeconds);
            if (Islands.IsValidIndex(Event.AudioIslandIndex))
            {
                bHasAudio = true;
                AudioStart = FMath::Min(AudioStart, N10SpeechIslandStartSec(Islands[Event.AudioIslandIndex]));
                AudioEnd = FMath::Max(AudioEnd, N10SpeechIslandEndSec(Islands[Event.AudioIslandIndex], ObservedAudioEndSec));
            }
        }
        if (!bHasAudio)
        {
            AudioStart = CommitStart;
            AudioEnd = CommitEnd;
        }
        int32 LandmarkCount = 0;
        int32 MatchedCount = 0;
        int32 Usable50 = 0;
        double AbsDeltaMs = 0.0;
        for (const FAU37AnchorMatch& A : AnchorRows)
        {
            if (!RegionEvents.count(A.EventIndex))
            {
                continue;
            }
            ++LandmarkCount;
            if (A.bMatched)
            {
                ++MatchedCount;
                if (A.BufferedUsefulLeadSec >= 0.050f)
                {
                    ++Usable50;
                }
                AbsDeltaMs += std::fabs(A.DeltaSec * 1000.0f);
            }
        }
        const float PlanDur = FMath::Max(0.0f, PlanEnd - PlanStart);
        const float AudioDur = FMath::Max(0.0f, AudioEnd - AudioStart);
        const float CommitDur = FMath::Max(0.0f, CommitEnd - CommitStart);
        Out << CsvEscape(Track.LineID.ToString().ToStdString()) << ','
            << CsvEscape(std::string(RegionType)) << ',' << RegionIndex << ',' << FirstEvent << ',' << LastEvent << ',' << (LastArrayIndex - FirstArrayIndex + 1) << ','
            << PlanStart << ',' << PlanEnd << ',' << PlanDur << ','
            << AudioStart << ',' << AudioEnd << ',' << AudioDur << ',' << (PlanDur > 0.001f ? AudioDur / PlanDur : 0.0f) << ','
            << CommitStart << ',' << CommitEnd << ',' << CommitDur << ','
            << ((CommitStart - AudioStart) * 1000.0f) << ',' << ((AudioEnd - CommitEnd) * 1000.0f) << ','
            << LandmarkCount << ',' << MatchedCount << ',' << Usable50 << ',' << (MatchedCount > 0 ? AbsDeltaMs / MatchedCount : 0.0) << "\n";
    };

    int32 RegionStart = 0;
    int32 RegionIndex = 0;
    while (RegionStart < Track.Events.Num())
    {
        const int32 TextIsland = Track.Events[RegionStart].TextIslandIndex;
        int32 RegionEnd = RegionStart;
        while (RegionEnd + 1 < Track.Events.Num() && Track.Events[RegionEnd + 1].TextIslandIndex == TextIsland)
        {
            ++RegionEnd;
        }
        AppendRange("punctuation", RegionIndex++, RegionStart, RegionEnd);
        RegionStart = RegionEnd + 1;
    }
    constexpr int32 SyntheticRegionEventCount = 8;
    for (int32 Start = 0, SyntheticIndex = 0; Start < Track.Events.Num(); Start += SyntheticRegionEventCount, ++SyntheticIndex)
    {
        AppendRange("synthetic", SyntheticIndex, Start, FMath::Min(Start + SyntheticRegionEventCount - 1, Track.Events.Num() - 1));
    }
}

static void WriteN11LandmarkPacingOpportunitiesCsv(
    const fs::path& EventPath,
    const fs::path& SummaryPath,
    const FOffgridAITextVisemePlan& Plan,
    const FOffgridAIAlignedVisemeTrack& Track,
    const TArray<FOffgridAIAudioLandmarkCandidate>& StreamingLandmarks,
    float PrerollSec)
{
    const auto Rows = BuildAU37AnchorMatches(Plan, Track, StreamingLandmarks, PrerollSec);
    std::ofstream EventOut(EventPath);
    EventOut << "LineID,EventIndex,TextIslandIndex,ProsodyGroupIndex,PoseID,SourceWord,LandmarkClass,PlannedCenterSec,CommittedCenterSec,Matched,CandidateTimeSec,DeltaFromPlannedMs,DeltaFromCommittedMs,Confidence,BufferedUsefulLeadMs,UsableLead25,UsableLead50\n";
    int32 LandmarkCount = 0;
    int32 MatchedCount = 0;
    int32 Usable25 = 0;
    int32 Usable50 = 0;
    double AbsDeltaCommittedMs = 0.0;
    float FirstPlan = 0.0f, LastPlan = 0.0f, FirstAudio = 0.0f, LastAudio = 0.0f;
    bool bHaveFirstMatched = false;
    for (const FAU37AnchorMatch& Row : Rows)
    {
        ++LandmarkCount;
        if (Row.bMatched)
        {
            ++MatchedCount;
            if (Row.BufferedUsefulLeadSec >= 0.025f) { ++Usable25; }
            if (Row.BufferedUsefulLeadSec >= 0.050f) { ++Usable50; }
            AbsDeltaCommittedMs += std::fabs(Row.DeltaSec * 1000.0f);
            if (!bHaveFirstMatched || Row.PlannedCenterSec < FirstPlan)
            {
                FirstPlan = Row.PlannedCenterSec;
                FirstAudio = Row.CandidateTimeSec;
            }
            if (!bHaveFirstMatched || Row.PlannedCenterSec > LastPlan)
            {
                LastPlan = Row.PlannedCenterSec;
                LastAudio = Row.CandidateTimeSec;
            }
            bHaveFirstMatched = true;
        }
        EventOut << CsvEscape(Track.LineID.ToString().ToStdString()) << ',' << Row.EventIndex << ',' << Row.IslandIndex << ',' << Row.GroupIndex << ','
            << CsvEscape(Row.PoseID.ToString().ToStdString()) << ',' << CsvEscape(Row.SourceWord.ToStdString()) << ',' << CsvEscape(Row.LandmarkClass) << ','
            << Row.PlannedCenterSec << ',' << Row.CommittedCenterSec << ',' << (Row.bMatched ? 1 : 0) << ',';
        if (Row.bMatched)
        {
            EventOut << Row.CandidateTimeSec << ',' << ((Row.CandidateTimeSec - Row.PlannedCenterSec) * 1000.0f) << ',' << (Row.DeltaSec * 1000.0f) << ','
                << Row.CandidateConfidence << ',' << (Row.BufferedUsefulLeadSec * 1000.0f) << ',' << (Row.BufferedUsefulLeadSec >= 0.025f ? 1 : 0) << ',' << (Row.BufferedUsefulLeadSec >= 0.050f ? 1 : 0) << "\n";
        }
        else
        {
            EventOut << ",,,,,,\n";
        }
    }
    const float AnchorPlanSpan = bHaveFirstMatched ? FMath::Max(0.0f, LastPlan - FirstPlan) : 0.0f;
    const float AnchorAudioSpan = bHaveFirstMatched ? FMath::Max(0.0f, LastAudio - FirstAudio) : 0.0f;
    std::ofstream SummaryOut(SummaryPath);
    SummaryOut << "LineID,LandmarkEventCount,MatchedLandmarkCount,MatchRate,UsableLead25Count,UsableLead50Count,UsableLead50Rate,MeanAbsCommittedDeltaMs,AnchorPlanSpanSec,AnchorAudioSpanSec,AnchorSpanScale\n";
    SummaryOut << CsvEscape(Track.LineID.ToString().ToStdString()) << ',' << LandmarkCount << ',' << MatchedCount << ','
        << (LandmarkCount > 0 ? static_cast<double>(MatchedCount) / LandmarkCount : 0.0) << ',' << Usable25 << ',' << Usable50 << ','
        << (LandmarkCount > 0 ? static_cast<double>(Usable50) / LandmarkCount : 0.0) << ','
        << (MatchedCount > 0 ? AbsDeltaCommittedMs / MatchedCount : 0.0) << ','
        << AnchorPlanSpan << ',' << AnchorAudioSpan << ',' << (AnchorPlanSpan > 0.001f ? AnchorAudioSpan / AnchorPlanSpan : 0.0f) << "\n";
}

static void WriteN11PacingSummaryCsv(
    const fs::path& Path,
    const FOffgridAITextVisemePlan& Plan,
    const FOffgridAIAlignedVisemeTrack& Track,
    const TArray<FOffgridAIStreamingSpeechIsland>& Islands,
    const TArray<FOffgridAIAudioLandmarkCandidate>& StreamingLandmarks,
    float ObservedAudioEndSec,
    float PrerollSec)
{
    const float PlannedActiveSec = FMath::Max(Plan.EstimatedDurationSeconds, 0.001f);
    const float ObservedActiveSec = N11ObservedActiveSec(Islands, ObservedAudioEndSec);
    float TotalPauseSec = 0.0f;
    float PrevEnd = -1.0f;
    std::vector<float> Durations;
    for (const FOffgridAIStreamingSpeechIsland& Island : Islands)
    {
        if (!Island.bStarted)
        {
            continue;
        }
        const float Start = N10SpeechIslandStartSec(Island);
        const float End = N10SpeechIslandEndSec(Island, ObservedAudioEndSec);
        if (PrevEnd >= 0.0f)
        {
            TotalPauseSec += FMath::Max(0.0f, Start - PrevEnd);
        }
        PrevEnd = End;
        Durations.push_back(FMath::Max(0.0f, End - Start));
    }
    double Mean = 0.0;
    for (float V : Durations) { Mean += V; }
    Mean = Durations.empty() ? 0.0 : Mean / Durations.size();
    double Var = 0.0;
    for (float V : Durations) { Var += (V - Mean) * (V - Mean); }
    Var = Durations.empty() ? 0.0 : Var / Durations.size();

    const auto AnchorRows = BuildAU37AnchorMatches(Plan, Track, StreamingLandmarks, PrerollSec);
    int32 LandmarkCount = 0;
    int32 MatchedCount = 0;
    int32 Usable50 = 0;
    double AbsDeltaMs = 0.0;
    for (const FAU37AnchorMatch& A : AnchorRows)
    {
        ++LandmarkCount;
        if (A.bMatched)
        {
            ++MatchedCount;
            if (A.BufferedUsefulLeadSec >= 0.050f) { ++Usable50; }
            AbsDeltaMs += std::fabs(A.DeltaSec * 1000.0f);
        }
    }

    std::ofstream Out(Path);
    Out << "LineID,PlannedActiveSec,ObservedActiveSec,ObservedToPlannedScale,SpeechIslandCount,TotalPauseSec,MeanSpeechIslandSec,SpeechIslandDurationCv,EventCount,LandmarkEventCount,MatchedLandmarkCount,LandmarkMatchRate,UsableLead50Count,MeanAbsLandmarkDeltaMs\n";
    Out << CsvEscape(Track.LineID.ToString().ToStdString()) << ',' << PlannedActiveSec << ',' << ObservedActiveSec << ',' << (ObservedActiveSec / PlannedActiveSec) << ','
        << Durations.size() << ',' << TotalPauseSec << ',' << Mean << ',' << (Mean > 0.001 ? std::sqrt(Var) / Mean : 0.0) << ','
        << Track.Events.Num() << ',' << LandmarkCount << ',' << MatchedCount << ',' << (LandmarkCount > 0 ? static_cast<double>(MatchedCount) / LandmarkCount : 0.0) << ','
        << Usable50 << ',' << (MatchedCount > 0 ? AbsDeltaMs / MatchedCount : 0.0) << "\n";
}


static void WriteN12PacingSummaryCsv(
    const fs::path& Path,
    const FOffgridAITextVisemePlan& Plan,
    const FOffgridAIAlignedVisemeTrack& Track,
    const TArray<FOffgridAIStreamingSpeechIsland>& Islands,
    float ObservedAudioEndSec);
static void WriteN12RegionDurationPacingSummaryCsv(
    const fs::path& Path,
    const FOffgridAITextVisemePlan& Plan,
    const FOffgridAIAlignedVisemeTrack& Track,
    const TArray<FOffgridAIStreamingSpeechIsland>& Islands,
    float ObservedAudioEndSec);
static void WriteN12RegionDurationPacingEventsCsv(
    const fs::path& Path,
    const FOffgridAITextVisemePlan& Plan,
    const FOffgridAIAlignedVisemeTrack& Track,
    const TArray<FOffgridAIStreamingSpeechIsland>& Islands,
    float ObservedAudioEndSec);
static void WriteN12RegionDurationPacingLineSummaryCsv(
    const fs::path& Path,
    const FOffgridAITextVisemePlan& Plan,
    const FOffgridAIAlignedVisemeTrack& Track,
    const TArray<FOffgridAIStreamingSpeechIsland>& Islands,
    float ObservedAudioEndSec);
static void WriteN11PacingOpportunityDiagnostics(
    const fs::path& CaseOutDir,
    const FOffgridAITextVisemePlan& Plan,
    const FOffgridAIAlignedVisemeTrack& Track,
    const TArray<FOffgridAIStreamingSpeechIsland>& Islands,
    const FWavData& Wav,
    int ChunkMs,
    float ObservedAudioEndSec,
    float PrerollSec)
{
    (void)Wav;
    (void)ChunkMs;
    (void)PrerollSec;
    WriteN12PacingSummaryCsv(CaseOutDir / "timing_pacing_summary.csv", Plan, Track, Islands, ObservedAudioEndSec);
    WriteN11OccupancyPacingMetricsCsv(CaseOutDir / "timing_occupancy_modulation.csv", Plan, Track, Islands, ObservedAudioEndSec);
    WriteN12RegionDurationPacingSummaryCsv(CaseOutDir / "timing_region_duration_pacing_summary.csv", Plan, Track, Islands, ObservedAudioEndSec);
    WriteN12RegionDurationPacingEventsCsv(CaseOutDir / "timing_region_duration_pacing_events.csv", Plan, Track, Islands, ObservedAudioEndSec);
    WriteN12RegionDurationPacingLineSummaryCsv(CaseOutDir / "timing_region_duration_pacing_line_summary.csv", Plan, Track, Islands, ObservedAudioEndSec);
}



struct FN12RegionDurationCandidate
{
    std::string RegionType;
    int32 RegionIndex = 0;
    int32 FirstTrackIndex = INDEX_NONE;
    int32 LastTrackIndex = INDEX_NONE;
    int32 FirstEventIndex = INDEX_NONE;
    int32 LastEventIndex = INDEX_NONE;
    int32 EventCount = 0;
    float FirstPlanCenterSec = 0.0f;
    float LastPlanCenterSec = 0.0f;
    float PlanCenterSpanSec = 0.0f;
    float AudioStartSec = 0.0f;
    float AudioEndSec = 0.0f;
    float AudioDurationSec = 0.0f;
    float CurrentFirstCenterSec = 0.0f;
    float CurrentLastCenterSec = 0.0f;
    float CurrentCenterSpanSec = 0.0f;
    int32 AudioIslandCount = 0;
    bool bReliable = false;
};

static FN12RegionDurationCandidate N12BuildRegionCandidate(
    const FOffgridAITextVisemePlan& Plan,
    const FOffgridAIAlignedVisemeTrack& Track,
    const TArray<FOffgridAIStreamingSpeechIsland>& Islands,
    float ObservedAudioEndSec,
    const std::string& RegionType,
    int32 RegionIndex,
    int32 FirstTrackIndex,
    int32 LastTrackIndex)
{
    FN12RegionDurationCandidate R;
    R.RegionType = RegionType;
    R.RegionIndex = RegionIndex;
    R.FirstTrackIndex = FirstTrackIndex;
    R.LastTrackIndex = LastTrackIndex;
    if (!Track.Events.IsValidIndex(FirstTrackIndex) || !Track.Events.IsValidIndex(LastTrackIndex) || LastTrackIndex < FirstTrackIndex)
    {
        return R;
    }

    R.FirstEventIndex = Track.Events[FirstTrackIndex].EventIndex;
    R.LastEventIndex = Track.Events[LastTrackIndex].EventIndex;
    R.EventCount = LastTrackIndex - FirstTrackIndex + 1;
    R.FirstPlanCenterSec = N11PlanCenterSec(Plan, R.FirstEventIndex);
    R.LastPlanCenterSec = N11PlanCenterSec(Plan, R.LastEventIndex);
    R.PlanCenterSpanSec = FMath::Max(0.0f, R.LastPlanCenterSec - R.FirstPlanCenterSec);
    R.CurrentFirstCenterSec = Track.Events[FirstTrackIndex].FinalRenderCenterSeconds;
    R.CurrentLastCenterSec = Track.Events[LastTrackIndex].FinalRenderCenterSeconds;
    R.CurrentCenterSpanSec = FMath::Max(0.0f, R.CurrentLastCenterSec - R.CurrentFirstCenterSec);

    float AudioStart = std::numeric_limits<float>::max();
    float AudioEnd = 0.0f;
    std::set<int32> AudioIslandIds;
    for (int32 I = FirstTrackIndex; I <= LastTrackIndex; ++I)
    {
        const FOffgridAIAlignedVisemeEvent& E = Track.Events[I];
        if (Islands.IsValidIndex(E.AudioIslandIndex))
        {
            AudioIslandIds.insert(E.AudioIslandIndex);
            AudioStart = FMath::Min(AudioStart, N10SpeechIslandStartSec(Islands[E.AudioIslandIndex]));
            AudioEnd = FMath::Max(AudioEnd, N10SpeechIslandEndSec(Islands[E.AudioIslandIndex], ObservedAudioEndSec));
        }
    }
    R.AudioIslandCount = static_cast<int32>(AudioIslandIds.size());
    if (R.AudioIslandCount > 0)
    {
        R.AudioStartSec = AudioStart;
        R.AudioEndSec = AudioEnd;
    }
    else
    {
        R.AudioStartSec = R.CurrentFirstCenterSec;
        R.AudioEndSec = R.CurrentLastCenterSec;
    }
    R.AudioDurationSec = FMath::Max(0.0f, R.AudioEndSec - R.AudioStartSec);
    const float Scale = R.PlanCenterSpanSec > 0.001f ? R.AudioDurationSec / R.PlanCenterSpanSec : 0.0f;
    R.bReliable = R.EventCount >= 2
        && R.PlanCenterSpanSec >= 0.120f
        && R.AudioDurationSec >= 0.120f
        && Scale >= 0.45f
        && Scale <= 2.25f
        && R.AudioIslandCount > 0;
    return R;
}

static std::vector<FN12RegionDurationCandidate> N12BuildRegionCandidates(
    const FOffgridAITextVisemePlan& Plan,
    const FOffgridAIAlignedVisemeTrack& Track,
    const TArray<FOffgridAIStreamingSpeechIsland>& Islands,
    float ObservedAudioEndSec)
{
    std::vector<FN12RegionDurationCandidate> Regions;
    int32 RegionStart = 0;
    int32 RegionIndex = 0;
    while (RegionStart < Track.Events.Num())
    {
        const int32 TextIsland = Track.Events[RegionStart].TextIslandIndex;
        int32 RegionEnd = RegionStart;
        while (RegionEnd + 1 < Track.Events.Num() && Track.Events[RegionEnd + 1].TextIslandIndex == TextIsland)
        {
            ++RegionEnd;
        }
        Regions.push_back(N12BuildRegionCandidate(Plan, Track, Islands, ObservedAudioEndSec, "punctuation", RegionIndex++, RegionStart, RegionEnd));
        RegionStart = RegionEnd + 1;
    }

    constexpr int32 SyntheticRegionEventCount = 8;
    for (int32 Start = 0, SyntheticIndex = 0; Start < Track.Events.Num(); Start += SyntheticRegionEventCount, ++SyntheticIndex)
    {
        Regions.push_back(N12BuildRegionCandidate(Plan, Track, Islands, ObservedAudioEndSec, "synthetic8", SyntheticIndex, Start, FMath::Min(Start + SyntheticRegionEventCount - 1, Track.Events.Num() - 1)));
    }
    return Regions;
}

static float N12RegionPacedCenterSec(
    const FN12RegionDurationCandidate& R,
    const FOffgridAITextVisemePlan& Plan,
    int32 EventIndex)
{
    if (!R.bReliable)
    {
        return 0.0f;
    }
    if (R.PlanCenterSpanSec <= 0.001f)
    {
        return (R.AudioStartSec + R.AudioEndSec) * 0.5f;
    }
    const float PlanCenter = N11PlanCenterSec(Plan, EventIndex);
    const float Alpha = FMath::Clamp((PlanCenter - R.FirstPlanCenterSec) / R.PlanCenterSpanSec, 0.0f, 1.0f);
    return R.AudioStartSec + Alpha * R.AudioDurationSec;
}

static void WriteN12RegionDurationPacingSummaryCsv(
    const fs::path& Path,
    const FOffgridAITextVisemePlan& Plan,
    const FOffgridAIAlignedVisemeTrack& Track,
    const TArray<FOffgridAIStreamingSpeechIsland>& Islands,
    float ObservedAudioEndSec)
{
    std::ofstream Out(Path);
    Out << "LineID,RegionType,RegionIndex,Reliable,FirstEventIndex,LastEventIndex,EventCount,AudioIslandCount,FirstPlanCenterSec,LastPlanCenterSec,PlanCenterSpanSec,AudioStartSec,AudioEndSec,AudioDurationSec,ObservedToPlanCenterScale,CurrentFirstCenterSec,CurrentLastCenterSec,CurrentCenterSpanSec,CurrentToAudioSpanScale,CurrentLeadingGapMs,CurrentTrailingGapMs,MeanAbsRegionRetimingMs,MaxAbsRegionRetimingMs,CurrentPauseCenterCount,RegionPacedPauseCenterCount,CurrentMeanDistanceToSpeechMs,RegionPacedMeanDistanceToSpeechMs\n";
    const std::vector<FN12RegionDurationCandidate> Regions = N12BuildRegionCandidates(Plan, Track, Islands, ObservedAudioEndSec);
    for (const FN12RegionDurationCandidate& R : Regions)
    {
        double AbsRetimingMs = 0.0;
        float MaxAbsRetimingMs = 0.0f;
        int32 CurrentPauseCount = 0;
        int32 RegionPauseCount = 0;
        double CurrentDistanceMs = 0.0;
        double RegionDistanceMs = 0.0;
        for (int32 I = R.FirstTrackIndex; I <= R.LastTrackIndex && Track.Events.IsValidIndex(I); ++I)
        {
            const FOffgridAIAlignedVisemeEvent& E = Track.Events[I];
            const float RegionCenter = N12RegionPacedCenterSec(R, Plan, E.EventIndex);
            if (R.bReliable)
            {
                const float DeltaMs = (RegionCenter - E.FinalRenderCenterSeconds) * 1000.0f;
                AbsRetimingMs += std::fabs(DeltaMs);
                MaxAbsRetimingMs = FMath::Max(MaxAbsRetimingMs, std::fabs(DeltaMs));
            }
            const bool bCurrentInside = IsCenterInsideSpeechIsland(Islands, E.FinalRenderCenterSeconds, ObservedAudioEndSec, 0.0f, 0.0f);
            const float CurrentDist = bCurrentInside ? 0.0f : DistanceToNearestSpeechIslandMs(Islands, E.FinalRenderCenterSeconds, ObservedAudioEndSec);
            CurrentDistanceMs += CurrentDist;
            if (!bCurrentInside) { ++CurrentPauseCount; }
            if (R.bReliable)
            {
                const bool bRegionInside = IsCenterInsideSpeechIsland(Islands, RegionCenter, ObservedAudioEndSec, 0.0f, 0.0f);
                const float RegionDist = bRegionInside ? 0.0f : DistanceToNearestSpeechIslandMs(Islands, RegionCenter, ObservedAudioEndSec);
                RegionDistanceMs += RegionDist;
                if (!bRegionInside) { ++RegionPauseCount; }
            }
        }
        const double Denom = R.EventCount > 0 ? static_cast<double>(R.EventCount) : 1.0;
        Out << CsvEscape(Track.LineID.ToString().ToStdString()) << ','
            << CsvEscape(R.RegionType) << ',' << R.RegionIndex << ',' << (R.bReliable ? 1 : 0) << ','
            << R.FirstEventIndex << ',' << R.LastEventIndex << ',' << R.EventCount << ',' << R.AudioIslandCount << ','
            << R.FirstPlanCenterSec << ',' << R.LastPlanCenterSec << ',' << R.PlanCenterSpanSec << ','
            << R.AudioStartSec << ',' << R.AudioEndSec << ',' << R.AudioDurationSec << ','
            << (R.PlanCenterSpanSec > 0.001f ? R.AudioDurationSec / R.PlanCenterSpanSec : 0.0f) << ','
            << R.CurrentFirstCenterSec << ',' << R.CurrentLastCenterSec << ',' << R.CurrentCenterSpanSec << ','
            << (R.AudioDurationSec > 0.001f ? R.CurrentCenterSpanSec / R.AudioDurationSec : 0.0f) << ','
            << ((R.CurrentFirstCenterSec - R.AudioStartSec) * 1000.0f) << ',' << ((R.AudioEndSec - R.CurrentLastCenterSec) * 1000.0f) << ','
            << (R.bReliable ? AbsRetimingMs / Denom : 0.0) << ',' << (R.bReliable ? MaxAbsRetimingMs : 0.0f) << ','
            << CurrentPauseCount << ',' << (R.bReliable ? RegionPauseCount : 0) << ','
            << (CurrentDistanceMs / Denom) << ',' << (R.bReliable ? RegionDistanceMs / Denom : 0.0) << "\n";
    }
}

static void WriteN12RegionDurationPacingEventsCsv(
    const fs::path& Path,
    const FOffgridAITextVisemePlan& Plan,
    const FOffgridAIAlignedVisemeTrack& Track,
    const TArray<FOffgridAIStreamingSpeechIsland>& Islands,
    float ObservedAudioEndSec)
{
    std::ofstream Out(Path);
    Out << "LineID,RegionType,RegionIndex,Reliable,EventIndex,PoseID,SourceWord,PlanCenterSec,CurrentCenterSec,RegionPacedCenterSec,RegionRetimingDeltaMs,CurrentInsideSpeech,RegionPacedInsideSpeech,CurrentDistanceToSpeechMs,RegionPacedDistanceToSpeechMs\n";
    const std::vector<FN12RegionDurationCandidate> Regions = N12BuildRegionCandidates(Plan, Track, Islands, ObservedAudioEndSec);
    for (const FN12RegionDurationCandidate& R : Regions)
    {
        for (int32 I = R.FirstTrackIndex; I <= R.LastTrackIndex && Track.Events.IsValidIndex(I); ++I)
        {
            const FOffgridAIAlignedVisemeEvent& E = Track.Events[I];
            const float PlanCenter = N11PlanCenterSec(Plan, E.EventIndex);
            const float RegionCenter = N12RegionPacedCenterSec(R, Plan, E.EventIndex);
            const bool bCurrentInside = IsCenterInsideSpeechIsland(Islands, E.FinalRenderCenterSeconds, ObservedAudioEndSec, 0.0f, 0.0f);
            const float CurrentDist = bCurrentInside ? 0.0f : DistanceToNearestSpeechIslandMs(Islands, E.FinalRenderCenterSeconds, ObservedAudioEndSec);
            bool bRegionInside = false;
            float RegionDist = 0.0f;
            if (R.bReliable)
            {
                bRegionInside = IsCenterInsideSpeechIsland(Islands, RegionCenter, ObservedAudioEndSec, 0.0f, 0.0f);
                RegionDist = bRegionInside ? 0.0f : DistanceToNearestSpeechIslandMs(Islands, RegionCenter, ObservedAudioEndSec);
            }
            Out << CsvEscape(Track.LineID.ToString().ToStdString()) << ','
                << CsvEscape(R.RegionType) << ',' << R.RegionIndex << ',' << (R.bReliable ? 1 : 0) << ','
                << E.EventIndex << ',' << CsvEscape(E.PoseID.ToString().ToStdString()) << ',' << CsvEscape(E.SourceWord.ToStdString()) << ','
                << PlanCenter << ',' << E.FinalRenderCenterSeconds << ',' << (R.bReliable ? RegionCenter : 0.0f) << ','
                << (R.bReliable ? (RegionCenter - E.FinalRenderCenterSeconds) * 1000.0f : 0.0f) << ','
                << (bCurrentInside ? 1 : 0) << ',' << (R.bReliable && bRegionInside ? 1 : 0) << ','
                << CurrentDist << ',' << (R.bReliable ? RegionDist : 0.0f) << "\n";
        }
    }
}

static void WriteN12RegionDurationPacingLineSummaryCsv(
    const fs::path& Path,
    const FOffgridAITextVisemePlan& Plan,
    const FOffgridAIAlignedVisemeTrack& Track,
    const TArray<FOffgridAIStreamingSpeechIsland>& Islands,
    float ObservedAudioEndSec)
{
    const std::vector<FN12RegionDurationCandidate> Regions = N12BuildRegionCandidates(Plan, Track, Islands, ObservedAudioEndSec);
    int32 ReliableRegionCount = 0;
    int32 ReliableEventCount = 0;
    int32 CurrentPauseCount = 0;
    int32 RegionPauseCount = 0;
    double CurrentDistanceMs = 0.0;
    double RegionDistanceMs = 0.0;
    double AbsRetimingMs = 0.0;
    float MaxAbsRetimingMs = 0.0f;
    double ScaleSum = 0.0;
    double ScaleSqSum = 0.0;
    int32 ScaleCount = 0;
    for (const FN12RegionDurationCandidate& R : Regions)
    {
        if (!R.bReliable)
        {
            continue;
        }
        ++ReliableRegionCount;
        const float Scale = R.PlanCenterSpanSec > 0.001f ? R.AudioDurationSec / R.PlanCenterSpanSec : 0.0f;
        ScaleSum += Scale;
        ScaleSqSum += Scale * Scale;
        ++ScaleCount;
        for (int32 I = R.FirstTrackIndex; I <= R.LastTrackIndex && Track.Events.IsValidIndex(I); ++I)
        {
            const FOffgridAIAlignedVisemeEvent& E = Track.Events[I];
            ++ReliableEventCount;
            const bool bCurrentInside = IsCenterInsideSpeechIsland(Islands, E.FinalRenderCenterSeconds, ObservedAudioEndSec, 0.0f, 0.0f);
            const float CurrentDist = bCurrentInside ? 0.0f : DistanceToNearestSpeechIslandMs(Islands, E.FinalRenderCenterSeconds, ObservedAudioEndSec);
            CurrentDistanceMs += CurrentDist;
            if (!bCurrentInside) { ++CurrentPauseCount; }
            const float RegionCenter = N12RegionPacedCenterSec(R, Plan, E.EventIndex);
            const bool bRegionInside = IsCenterInsideSpeechIsland(Islands, RegionCenter, ObservedAudioEndSec, 0.0f, 0.0f);
            const float RegionDist = bRegionInside ? 0.0f : DistanceToNearestSpeechIslandMs(Islands, RegionCenter, ObservedAudioEndSec);
            RegionDistanceMs += RegionDist;
            if (!bRegionInside) { ++RegionPauseCount; }
            const float DeltaMs = (RegionCenter - E.FinalRenderCenterSeconds) * 1000.0f;
            AbsRetimingMs += std::fabs(DeltaMs);
            MaxAbsRetimingMs = FMath::Max(MaxAbsRetimingMs, std::fabs(DeltaMs));
        }
    }
    const double EventDenom = ReliableEventCount > 0 ? static_cast<double>(ReliableEventCount) : 1.0;
    const double ScaleMean = ScaleCount > 0 ? ScaleSum / ScaleCount : 0.0;
    const double ScaleVar = ScaleCount > 0 ? FMath::Max(0.0, ScaleSqSum / ScaleCount - ScaleMean * ScaleMean) : 0.0;
    std::ofstream Out(Path);
    Out << "LineID,RegionCount,ReliableRegionCount,ReliableEventCount,CurrentPauseCenterCount,RegionPacedPauseCenterCount,CurrentMeanDistanceToSpeechMs,RegionPacedMeanDistanceToSpeechMs,MeanAbsRegionRetimingMs,MaxAbsRegionRetimingMs,MeanRegionScale,RegionScaleCv\n";
    Out << CsvEscape(Track.LineID.ToString().ToStdString()) << ',' << Regions.size() << ',' << ReliableRegionCount << ',' << ReliableEventCount << ','
        << CurrentPauseCount << ',' << RegionPauseCount << ',' << (CurrentDistanceMs / EventDenom) << ',' << (RegionDistanceMs / EventDenom) << ','
        << (AbsRetimingMs / EventDenom) << ',' << MaxAbsRetimingMs << ',' << ScaleMean << ',' << (ScaleMean > 0.001 ? std::sqrt(ScaleVar) / ScaleMean : 0.0) << "\n";
}

static void WriteN12PacingSummaryCsv(
    const fs::path& Path,
    const FOffgridAITextVisemePlan& Plan,
    const FOffgridAIAlignedVisemeTrack& Track,
    const TArray<FOffgridAIStreamingSpeechIsland>& Islands,
    float ObservedAudioEndSec)
{
    const float PlannedActiveSec = FMath::Max(Plan.EstimatedDurationSeconds, 0.001f);
    const float ObservedActiveSec = N11ObservedActiveSec(Islands, ObservedAudioEndSec);
    float TotalPauseSec = 0.0f;
    float PrevEnd = -1.0f;
    std::vector<float> Durations;
    for (const FOffgridAIStreamingSpeechIsland& Island : Islands)
    {
        if (!Island.bStarted)
        {
            continue;
        }
        const float Start = N10SpeechIslandStartSec(Island);
        const float End = N10SpeechIslandEndSec(Island, ObservedAudioEndSec);
        if (PrevEnd >= 0.0f)
        {
            TotalPauseSec += FMath::Max(0.0f, Start - PrevEnd);
        }
        PrevEnd = End;
        Durations.push_back(FMath::Max(0.0f, End - Start));
    }
    double Mean = 0.0;
    for (float V : Durations) { Mean += V; }
    Mean = Durations.empty() ? 0.0 : Mean / Durations.size();
    double Var = 0.0;
    for (float V : Durations) { Var += (V - Mean) * (V - Mean); }
    Var = Durations.empty() ? 0.0 : Var / Durations.size();

    std::ofstream Out(Path);
    Out << "LineID,PlannedActiveSec,ObservedActiveSec,ObservedToPlannedScale,SpeechIslandCount,TotalPauseSec,MeanSpeechIslandSec,SpeechIslandDurationCv,EventCount\n";
    Out << CsvEscape(Track.LineID.ToString().ToStdString()) << ',' << PlannedActiveSec << ',' << ObservedActiveSec << ',' << (ObservedActiveSec / PlannedActiveSec) << ','
        << Durations.size() << ',' << TotalPauseSec << ',' << Mean << ',' << (Mean > 0.001 ? std::sqrt(Var) / Mean : 0.0) << ','
        << Track.Events.Num() << "\n";
}
static void WriteScorecardCsv(
    const fs::path& Path,
    const std::string& CaseId,
    const std::string& Transcript,
    const FOffgridAITextVisemePlan& Plan,
    const FOffgridAIAlignedVisemeTrack& Track,
    const FSubmittedPoseDiagnostics& PoseDiagnostics,
    const FTimingDiagnostics& TimingDiagnostics,
    const TArray<FOffgridAIStreamingSpeechIsland>& SpeechIslands,
    float AudioDurationSec)
{
    std::vector<int> EventCommitCounts(static_cast<size_t>(Plan.Events.Num()), 0);
    int InvalidCommittedEventIndexCount = 0;
    for (const FOffgridAIAlignedVisemeEvent& E : Track.Events)
    {
        if (E.EventIndex >= 0 && E.EventIndex < Plan.Events.Num())
        {
            ++EventCommitCounts[static_cast<size_t>(E.EventIndex)];
        }
        else
        {
            ++InvalidCommittedEventIndexCount;
        }
    }

    int MissingCommitted = 0;
    int DuplicateCommittedEventCount = 0;
    int CoveredEventCount = 0;
    int MaxMissingEventIndexRun = 0;
    int CurrentMissingEventIndexRun = 0;
    for (int Count : EventCommitCounts)
    {
        if (Count <= 0)
        {
            ++MissingCommitted;
            ++CurrentMissingEventIndexRun;
            MaxMissingEventIndexRun = std::max(MaxMissingEventIndexRun, CurrentMissingEventIndexRun);
        }
        else
        {
            ++CoveredEventCount;
            CurrentMissingEventIndexRun = 0;
            if (Count > 1)
            {
                DuplicateCommittedEventCount += Count - 1;
            }
        }
    }

    int MonotonicViolations = 0;
    int OutOfOrderCommittedEventCount = 0;
    int EventsCenteredOutsideSpeechIslandCount = 0;
    float MaxDistanceOutsideSpeechIslandMs = 0.0f;
    int SoftIslandEndOverrunCount = 0;
    float MaxSoftIslandEndOverrunMs = 0.0f;
    int WeakPeakCount = 0;
    int ExplosiveWeakPeakCount = 0;
    int ExplosivePeakCount = 0;
    int VowelWeakPeakCount = 0;
    int VowelPeakCount = 0;
    int MissingPeakCount = 0;
    int LandmarkWeakPeakCount = 0;
    int LandmarkPeakCount = 0;
    int DominantWeakPeakCount = 0;
    int DominantPeakCount = 0;
    double DominantPeakRatioSum = 0.0;
    int DominantPeakRatioCount = 0;
    float DominantPeakRatioMin = 1.0f;
    float MinCommittedCenterGapMs = 0.0f;
    int CompressedEventGapCountUnder16ms = 0;
    int CompressedEventGapCountUnder33ms = 0;
    int MaxCompressedRunLengthUnder33ms = 0;
    int SameFamilyRepeatCountUnder120ms = 0;
    int SameFamilyRepeatUnprotectedCountUnder120ms = 0;
    int DenseIslandMaxEventCount = 0;
    double DenseIslandMaxEventsPerSec = 0.0;
    int PhraseStartEvaluatedCount = 0;
    int EarlyPhraseStartCount = 0;
    float MaxEarlyPhraseStartLeadMs = 0.0f;
    int SeverelyCompressedPhraseCount = 0;
    float MinPhraseDurationRatio = 1.0f;
    double PeakAbsErrorSum = 0.0;
    int PeakErrorCount = 0;
    double PeakWindowCenterAbsErrorSum = 0.0;
    int PeakWindowCenterCount = 0;
    double PeakWindowWidthSumSec = 0.0;
    int PeakWindowWidthCount = 0;
    double ExplosivePeakLeadSumMs = 0.0;
    double ExplosivePeakLeadAbsErrorSumMs = 0.0;
    int ExplosivePeakLeadCount = 0;
    double VowelPeakCenterAbsErrorSumMs = 0.0;
    int VowelPeakCenterCount = 0;
    float ScaleMax = 0.0f;
    int ScaleCapHitCount = 0;
    int NudgeProposedCount = 0;
    int NudgeAcceptedCount = 0;
    double NudgeAbsDeltaSumMs = 0.0;
    float MaxAbsNudgeDeltaMs = 0.0f;
    double CommitLeadSum = 0.0;
    int CommitLeadCount = 0;
    int CommitReasonWithinHorizonCount = 0;
    int CommitReasonAheadOfHorizonCount = 0;
    int CommitReasonEndOfStreamCount = 0;
    int CommitReasonFallbackCount = 0;
    int CommitReasonUnknownCount = 0;
    int CenterOrderRepairCount = 0;
    double CenterOrderRepairAbsSumMs = 0.0;
    float CenterOrderRepairMaxMs = 0.0f;

    for (int32 I = 1; I < Track.Events.Num(); ++I)
    {
        if (Track.Events[I].FinalRenderCenterSeconds <= Track.Events[I - 1].FinalRenderCenterSeconds)
        {
            ++MonotonicViolations;
        }
        if (Track.Events[I].EventIndex <= Track.Events[I - 1].EventIndex)
        {
            ++OutOfOrderCommittedEventCount;
        }
    }

    if (Track.Events.Num() > 1)
    {
        MinCommittedCenterGapMs = TNumericLimits<float>::Max();
        int CurrentCompressedRunLength = 1;
        for (int32 I = 1; I < Track.Events.Num(); ++I)
        {
            const float GapMs = FMath::Max(
                Track.Events[I].FinalRenderCenterSeconds - Track.Events[I - 1].FinalRenderCenterSeconds,
                0.0f) * 1000.0f;
            MinCommittedCenterGapMs = FMath::Min(MinCommittedCenterGapMs, GapMs);
            if (GapMs < 16.0f)
            {
                ++CompressedEventGapCountUnder16ms;
            }
            if (GapMs < 33.0f)
            {
                ++CompressedEventGapCountUnder33ms;
                ++CurrentCompressedRunLength;
            }
            else
            {
                MaxCompressedRunLengthUnder33ms = FMath::Max(MaxCompressedRunLengthUnder33ms, CurrentCompressedRunLength);
                CurrentCompressedRunLength = 1;
            }
        }
        MaxCompressedRunLengthUnder33ms = FMath::Max(MaxCompressedRunLengthUnder33ms, CurrentCompressedRunLength);
        if (MinCommittedCenterGapMs == TNumericLimits<float>::Max())
        {
            MinCommittedCenterGapMs = 0.0f;
        }

        for (int32 I = 0; I < Track.Events.Num(); ++I)
        {
            const FOffgridAIAlignedVisemeEvent& Curr = Track.Events[I];
            const std::string CurrFamily = LandmarkClassForPose(Curr.PoseID);
            if (CurrFamily == "OTHER")
            {
                continue;
            }
            for (int32 PrevI = I - 1; PrevI >= 0; --PrevI)
            {
                const FOffgridAIAlignedVisemeEvent& Prev = Track.Events[PrevI];
                if (Prev.TextIslandIndex != Curr.TextIslandIndex)
                {
                    break;
                }
                const float GapMs = (Curr.FinalRenderCenterSeconds - Prev.FinalRenderCenterSeconds) * 1000.0f;
                if (GapMs > 120.0f)
                {
                    break;
                }
                if (GapMs >= 0.0f && LandmarkClassForPose(Prev.PoseID) == CurrFamily)
                {
                    ++SameFamilyRepeatCountUnder120ms;
                    if (!Prev.bIsLandmark && !Curr.bIsLandmark)
                    {
                        ++SameFamilyRepeatUnprotectedCountUnder120ms;
                    }
                }
            }
        }
    }

    TMap<int32, TArray<const FOffgridAIAlignedVisemeEvent*>> DensityEventsByIsland;
    for (const FOffgridAIAlignedVisemeEvent& E : Track.Events)
    {
        DensityEventsByIsland.FindOrAdd(E.TextIslandIndex).Add(&E);
    }
    for (const TPair<int32, TArray<const FOffgridAIAlignedVisemeEvent*>>& Pair : DensityEventsByIsland)
    {
        const TArray<const FOffgridAIAlignedVisemeEvent*>& IslandEvents = Pair.Value;
        if (IslandEvents.Num() <= 0)
        {
            continue;
        }
        float IslandStartSec = TNumericLimits<float>::Max();
        float IslandEndSec = -TNumericLimits<float>::Max();
        for (const FOffgridAIAlignedVisemeEvent* E : IslandEvents)
        {
            IslandStartSec = FMath::Min(IslandStartSec, E->FinalRenderCenterSeconds);
            IslandEndSec = FMath::Max(IslandEndSec, E->FinalRenderCenterSeconds);
        }
        const float IslandSpanSec = FMath::Max(IslandEndSec - IslandStartSec, 0.001f);
        const double EventsPerSec = static_cast<double>(IslandEvents.Num()) / IslandSpanSec;
        if (EventsPerSec > DenseIslandMaxEventsPerSec)
        {
            DenseIslandMaxEventsPerSec = EventsPerSec;
            DenseIslandMaxEventCount = IslandEvents.Num();
        }
    }

    for (int32 I = 1; I < Track.Events.Num(); ++I)
    {
        const FOffgridAIAlignedVisemeEvent& Prev = Track.Events[I - 1];
        const FOffgridAIAlignedVisemeEvent& Curr = Track.Events[I];
        if (Prev.PhraseIndex == Curr.PhraseIndex)
        {
            continue;
        }

        const std::optional<float> NextIslandStart = FindNextSpeechIslandStartSec(
            SpeechIslands,
            Prev.FinalRenderCenterSeconds + 0.050f);
        if (!NextIslandStart)
        {
            continue;
        }

        constexpr float AllowedAnticipationSec = 0.075f;
        const float EarlyLeadSec = *NextIslandStart - Curr.FinalRenderCenterSeconds;
        if (EarlyLeadSec > 0.750f)
        {
            continue;
        }
        ++PhraseStartEvaluatedCount;
        if (EarlyLeadSec > AllowedAnticipationSec)
        {
            ++EarlyPhraseStartCount;
            MaxEarlyPhraseStartLeadMs = FMath::Max(MaxEarlyPhraseStartLeadMs, EarlyLeadSec * 1000.0f);
        }
    }

    TMap<int32, TArray<const FOffgridAIAlignedVisemeEvent*>> EventsByPhrase;
    for (const FOffgridAIAlignedVisemeEvent& E : Track.Events)
    {
        EventsByPhrase.FindOrAdd(E.PhraseIndex).Add(&E);
    }
    for (const TPair<int32, TArray<const FOffgridAIAlignedVisemeEvent*>>& Pair : EventsByPhrase)
    {
        const TArray<const FOffgridAIAlignedVisemeEvent*>& PhraseEvents = Pair.Value;
        if (PhraseEvents.Num() < 3)
        {
            continue;
        }

        float CommittedStartSec = TNumericLimits<float>::Max();
        float CommittedEndSec = -TNumericLimits<float>::Max();
        float TextStartSec = TNumericLimits<float>::Max();
        float TextEndSec = -TNumericLimits<float>::Max();
        for (const FOffgridAIAlignedVisemeEvent* Event : PhraseEvents)
        {
            if (!Event || !Plan.Events.IsValidIndex(Event->EventIndex))
            {
                continue;
            }
            CommittedStartSec = FMath::Min(CommittedStartSec, Event->FinalRenderCenterSeconds);
            CommittedEndSec = FMath::Max(CommittedEndSec, Event->FinalRenderCenterSeconds);
            TextStartSec = FMath::Min(TextStartSec, Event->TextDiagnosticCenterSeconds);
            TextEndSec = FMath::Max(TextEndSec, Event->TextDiagnosticCenterSeconds);
        }

        const float CommittedSpanSec = FMath::Max(CommittedEndSec - CommittedStartSec, 0.0f);
        const float TextSpanSec = FMath::Max(TextEndSec - TextStartSec, 0.001f);
        const float Ratio = CommittedSpanSec / TextSpanSec;
        MinPhraseDurationRatio = FMath::Min(MinPhraseDurationRatio, Ratio);
        if (Ratio < 0.12f)
        {
            ++SeverelyCompressedPhraseCount;
        }
    }

    for (const auto& E : Track.Events)
    {
        if (!IsCenterInsideSpeechIsland(SpeechIslands, E.FinalRenderCenterSeconds, AudioDurationSec, 0.050f, 0.050f))
        {
            ++EventsCenteredOutsideSpeechIslandCount;
            MaxDistanceOutsideSpeechIslandMs = FMath::Max(
                MaxDistanceOutsideSpeechIslandMs,
                DistanceToNearestSpeechIslandMs(SpeechIslands, E.FinalRenderCenterSeconds, AudioDurationSec));
        }
        const float SoftOverrunMs = LateDistancePastNearestSpeechIslandEndMs(SpeechIslands, E.FinalRenderCenterSeconds, AudioDurationSec);
        if (SoftOverrunMs > 160.0f)
        {
            ++SoftIslandEndOverrunCount;
            MaxSoftIslandEndOverrunMs = FMath::Max(MaxSoftIslandEndOverrunMs, SoftOverrunMs);
        }
        ScaleMax = std::max(ScaleMax, E.EffectiveSegmentScale);
        if (E.EffectiveSegmentScale >= 2.24f) { ++ScaleCapHitCount; }
        if (WasNudgeProposedForEvent(E)) { ++NudgeProposedCount; }
        if (WasNudgeAcceptedForEvent(E))
        {
            ++NudgeAcceptedCount;
            const float AbsDeltaMs = FMath::Abs(E.AppliedShiftSeconds) * 1000.0f;
            NudgeAbsDeltaSumMs += AbsDeltaMs;
            MaxAbsNudgeDeltaMs = FMath::Max(MaxAbsNudgeDeltaMs, AbsDeltaMs);
        }
        if (std::isfinite(E.CommitLeadSeconds))
        {
            CommitLeadSum += E.CommitLeadSeconds;
            ++CommitLeadCount;
        }
        const FString CommitReasonString = E.CommitReason.ToString();
        if (CommitReasonString == TEXT("within_horizon"))
        {
            ++CommitReasonWithinHorizonCount;
        }
        else if (CommitReasonString == TEXT("ahead_of_horizon_track_rebuild") || CommitReasonString == TEXT("assigned_audio_island"))
        {
            ++CommitReasonAheadOfHorizonCount;
        }
        else if (CommitReasonString == TEXT("end_of_stream") || CommitReasonString == TEXT("end_of_stream_flush"))
        {
            ++CommitReasonEndOfStreamCount;
        }
        else if (CommitReasonString == TEXT("fallback_or_missing_audio") || CommitReasonString == TEXT("late_publication"))
        {
            ++CommitReasonFallbackCount;
        }
        else
        {
            ++CommitReasonUnknownCount;
        }
        if (E.bCenterOrderRepaired)
        {
            ++CenterOrderRepairCount;
            const float RepairMs = FMath::Abs(E.CenterOrderRepairSeconds) * 1000.0f;
            CenterOrderRepairAbsSumMs += RepairMs;
            CenterOrderRepairMaxMs = FMath::Max(CenterOrderRepairMaxMs, RepairMs);
        }
        auto It = PoseDiagnostics.Peaks.find(E.EventIndex);
        if (It == PoseDiagnostics.Peaks.end() || It->second.PeakWeight < 0.0f)
        {
            ++MissingPeakCount;
            continue;
        }
        const FPeakInfo& Peak = It->second;
        if (It->second.PeakWeight < 0.20f)
        {
            ++WeakPeakCount;
        }
        if (E.bIsLandmark)
        {
            ++LandmarkPeakCount;
            if (Peak.PeakWeight < 0.42f)
            {
                ++LandmarkWeakPeakCount;
            }
        }
        if (E.bIsLandmark || E.Strength >= 0.78f)
        {
            ++DominantPeakCount;
            if (Peak.PeakWeight < 0.50f)
            {
                ++DominantWeakPeakCount;
            }
            const float PeakRatio = Peak.PeakWeight / FMath::Max(E.Strength, 0.001f);
            DominantPeakRatioMin = FMath::Min(DominantPeakRatioMin, PeakRatio);
            DominantPeakRatioSum += PeakRatio;
            ++DominantPeakRatioCount;
        }
        const float PeakWindowCenterSec = PeakWindowWeightedCenterTime(Peak);
        const float PeakWindowWidthSecValue = PeakWindowWidthSec(Peak);
        PeakAbsErrorSum += std::abs(Peak.PeakTime - E.FinalRenderCenterSeconds);
        ++PeakErrorCount;
        PeakWindowCenterAbsErrorSum += std::abs(PeakWindowCenterSec - E.FinalRenderCenterSeconds);
        ++PeakWindowCenterCount;
        PeakWindowWidthSumSec += PeakWindowWidthSecValue;
        ++PeakWindowWidthCount;

        if (IsExplosivePose(E.PoseID))
        {
            ++ExplosivePeakCount;
            if (Peak.PeakWeight < 0.20f)
            {
                ++ExplosiveWeakPeakCount;
            }
            const float ActualLeadMs = 1000.0f * (E.FinalRenderCenterSeconds - Peak.PeakWindowStart);
            const float DesiredLeadMs = 1000.0f * DesiredPeakHoldHalfSecondsForPose(E.PoseID);
            ExplosivePeakLeadSumMs += ActualLeadMs;
            ExplosivePeakLeadAbsErrorSumMs += std::abs(ActualLeadMs - DesiredLeadMs);
            ++ExplosivePeakLeadCount;
        }
        else if (IsVowelLikePose(E.PoseID) || IsGlidePose(E.PoseID))
        {
            ++VowelPeakCount;
            if (Peak.PeakWeight < 0.20f)
            {
                ++VowelWeakPeakCount;
            }
            VowelPeakCenterAbsErrorSumMs += 1000.0 * std::abs(PeakWindowCenterSec - E.FinalRenderCenterSeconds);
            ++VowelPeakCenterCount;
        }
    }

    const int HardInvariantFailureCount =
        MissingCommitted
        + DuplicateCommittedEventCount
        + InvalidCommittedEventIndexCount
        + OutOfOrderCommittedEventCount
        + MonotonicViolations;

    float RuntimeCommittedStartSec = std::numeric_limits<float>::max();
    float RuntimeCommittedEndSec = 0.0f;
    float RuntimePlannerIslandDurationSumSec = 0.0f;
    int RuntimePlannerIslandDurationCount = 0;
    std::set<int> RuntimeSeenTextIslands;
    for (const FOffgridAIAlignedVisemeEvent& E : Track.Events)
    {
        RuntimeCommittedStartSec = FMath::Min(RuntimeCommittedStartSec, E.RenderStartSeconds);
        RuntimeCommittedEndSec = FMath::Max(RuntimeCommittedEndSec, E.RenderEndSeconds);
        if (E.TextIslandIndex != INDEX_NONE && RuntimeSeenTextIslands.insert(E.TextIslandIndex).second)
        {
            RuntimePlannerIslandDurationSumSec += FMath::Max(E.PlannerIslandPredictedDurationSeconds, 0.0f);
            ++RuntimePlannerIslandDurationCount;
        }
    }
    if (Track.Events.Num() <= 0 || !FMath::IsFinite(RuntimeCommittedStartSec))
    {
        RuntimeCommittedStartSec = 0.0f;
        RuntimeCommittedEndSec = 0.0f;
    }
    const float RuntimeCommittedDurationSec = FMath::Max(RuntimeCommittedEndSec - RuntimeCommittedStartSec, 0.0f);
    const float RuntimePlannerIslandDurationTotalSec = RuntimePlannerIslandDurationSumSec;
    const float RuntimePlannerIslandDurationMeanSec = RuntimePlannerIslandDurationCount > 0
        ? RuntimePlannerIslandDurationSumSec / static_cast<float>(RuntimePlannerIslandDurationCount)
        : 0.0f;

    std::ofstream Out(Path);
    Out << "Metric,Value\n";
    Out << "CaseId," << CsvEscape(CaseId) << "\n";
    Out << "Transcript," << CsvEscape(Transcript) << "\n";
    Out << "PlannedEventCount," << Plan.Events.Num() << "\n";
    Out << "CommittedEventCount," << Track.Events.Num() << "\n";
    Out << "MissingCommittedEventCount," << MissingCommitted << "\n";
    Out << "DuplicateCommittedEventCount," << DuplicateCommittedEventCount << "\n";
    Out << "InvalidCommittedEventIndexCount," << InvalidCommittedEventIndexCount << "\n";
    Out << "OutOfOrderCommittedEventCount," << OutOfOrderCommittedEventCount << "\n";
    Out << "MonotonicityViolations," << MonotonicViolations << "\n";
    Out << "NonMonotonicCommittedCenterCount," << MonotonicViolations << "\n";
    Out << "EventIndexCoverageRate," << (Plan.Events.Num() > 0 ? static_cast<double>(CoveredEventCount) / Plan.Events.Num() : 0.0) << "\n";
    Out << "MaxMissingEventIndexRun," << MaxMissingEventIndexRun << "\n";
    Out << "HardInvariantFailureCount," << HardInvariantFailureCount << "\n";
    Out << "EventsCenteredOutsideSpeechIslandCount," << EventsCenteredOutsideSpeechIslandCount << "\n";
    Out << "EventsCenteredInsideSpeechIslandRate," << (Track.Events.Num() > 0 ? static_cast<double>(Track.Events.Num() - EventsCenteredOutsideSpeechIslandCount) / Track.Events.Num() : 0.0) << "\n";
    Out << "MaxDistanceOutsideSpeechIslandMs," << MaxDistanceOutsideSpeechIslandMs << "\n";
    Out << "SoftIslandEndOverrunCount," << SoftIslandEndOverrunCount << "\n";
    Out << "MaxSoftIslandEndOverrunMs," << MaxSoftIslandEndOverrunMs << "\n";
    Out << "MinCommittedCenterGapMs," << MinCommittedCenterGapMs << "\n";
    Out << "CenterOrderRepairCount," << CenterOrderRepairCount << "\n";
    Out << "CenterOrderRepairMeanMs," << (CenterOrderRepairCount > 0 ? CenterOrderRepairAbsSumMs / CenterOrderRepairCount : 0.0) << "\n";
    Out << "CenterOrderRepairMaxMs," << CenterOrderRepairMaxMs << "\n";
    Out << "CompressedEventGapCountUnder16ms," << CompressedEventGapCountUnder16ms << "\n";
    Out << "CompressedEventGapCountUnder33ms," << CompressedEventGapCountUnder33ms << "\n";
    Out << "MaxCompressedRunLengthUnder33ms," << MaxCompressedRunLengthUnder33ms << "\n";
    Out << "SameFamilyRepeatCountUnder120ms," << SameFamilyRepeatCountUnder120ms << "\n";
    Out << "SameFamilyRepeatUnprotectedCountUnder120ms," << SameFamilyRepeatUnprotectedCountUnder120ms << "\n";
    Out << "DenseIslandMaxEventCount," << DenseIslandMaxEventCount << "\n";
    Out << "DenseIslandMaxEventsPerSec," << DenseIslandMaxEventsPerSec << "\n";
    Out << "PhraseStartEvaluatedCount," << PhraseStartEvaluatedCount << "\n";
    Out << "EarlyPhraseStartCount," << EarlyPhraseStartCount << "\n";
    Out << "MaxEarlyPhraseStartLeadMs," << MaxEarlyPhraseStartLeadMs << "\n";
    Out << "SeverelyCompressedPhraseCount," << SeverelyCompressedPhraseCount << "\n";
    Out << "MinPhraseDurationRatio," << MinPhraseDurationRatio << "\n";
    Out << "MissingPeakCount," << MissingPeakCount << "\n";
    Out << "WeakPeakCount," << WeakPeakCount << "\n";
    Out << "WeakPeakRate," << (Track.Events.Num() > 0 ? static_cast<double>(WeakPeakCount) / Track.Events.Num() : 0.0) << "\n";
    Out << "LandmarkWeakPeakCount," << LandmarkWeakPeakCount << "\n";
    Out << "LandmarkWeakPeakRate," << (LandmarkPeakCount > 0 ? static_cast<double>(LandmarkWeakPeakCount) / LandmarkPeakCount : 0.0) << "\n";
    Out << "DominantWeakPeakCount," << DominantWeakPeakCount << "\n";
    Out << "DominantWeakPeakRate," << (DominantPeakCount > 0 ? static_cast<double>(DominantWeakPeakCount) / DominantPeakCount : 0.0) << "\n";
    Out << "DominantPeakRatioMin," << (DominantPeakRatioCount > 0 ? DominantPeakRatioMin : 0.0f) << "\n";
    Out << "DominantPeakRatioMean," << (DominantPeakRatioCount > 0 ? DominantPeakRatioSum / DominantPeakRatioCount : 0.0) << "\n";
    Out << "PeakTimingMAEMs," << (PeakErrorCount > 0 ? 1000.0 * PeakAbsErrorSum / PeakErrorCount : 0.0) << "\n";
    Out << "PeakWindowCenterMAEMs," << (PeakWindowCenterCount > 0 ? 1000.0 * PeakWindowCenterAbsErrorSum / PeakWindowCenterCount : 0.0) << "\n";
    Out << "PeakWindowWidthMeanMs," << (PeakWindowWidthCount > 0 ? 1000.0 * PeakWindowWidthSumSec / PeakWindowWidthCount : 0.0) << "\n";
    Out << "ExplosiveWeakPeakRate," << (ExplosivePeakCount > 0 ? static_cast<double>(ExplosiveWeakPeakCount) / ExplosivePeakCount : 0.0) << "\n";
    Out << "ExplosivePeakLeadMeanMs," << (ExplosivePeakLeadCount > 0 ? ExplosivePeakLeadSumMs / ExplosivePeakLeadCount : 0.0) << "\n";
    Out << "ExplosivePeakLeadErrorMAEMs," << (ExplosivePeakLeadCount > 0 ? ExplosivePeakLeadAbsErrorSumMs / ExplosivePeakLeadCount : 0.0) << "\n";
    Out << "VowelWeakPeakRate," << (VowelPeakCount > 0 ? static_cast<double>(VowelWeakPeakCount) / VowelPeakCount : 0.0) << "\n";
    Out << "VowelPeakCenterMAEMs," << (VowelPeakCenterCount > 0 ? VowelPeakCenterAbsErrorSumMs / VowelPeakCenterCount : 0.0) << "\n";
    Out << "EstimatedTextDurationSec," << Plan.EstimatedDurationSeconds << "\n";
    Out << "RuntimeCommittedDurationSec," << RuntimeCommittedDurationSec << "\n";
    Out << "RuntimeCommittedStartSec," << RuntimeCommittedStartSec << "\n";
    Out << "RuntimeCommittedEndSec," << RuntimeCommittedEndSec << "\n";
    Out << "RuntimePlannerIslandDurationTotalSec," << RuntimePlannerIslandDurationTotalSec << "\n";
    Out << "RuntimePlannerIslandDurationMeanSec," << RuntimePlannerIslandDurationMeanSec << "\n";
    Out << "RuntimePlannerIslandDurationCount," << RuntimePlannerIslandDurationCount << "\n";
    Out << "AudioDurationSec," << AudioDurationSec << "\n";
    Out << "EffectiveSegmentScaleMax," << ScaleMax << "\n";
    Out << "ScaleCapHitCount," << ScaleCapHitCount << "\n";
    Out << "ProposedNudgeCount," << NudgeProposedCount << "\n";
    Out << "AcceptedNudgeCount," << NudgeAcceptedCount << "\n";
    Out << "AcceptedNudgeAbsMeanMs," << (NudgeAcceptedCount > 0 ? NudgeAbsDeltaSumMs / NudgeAcceptedCount : 0.0) << "\n";
    Out << "AcceptedNudgeAbsMaxMs," << MaxAbsNudgeDeltaMs << "\n";
    Out << "CommitLeadMeanSec," << (CommitLeadCount > 0 ? CommitLeadSum / CommitLeadCount : 0.0) << "\n";
    Out << "CommitReason_WithinHorizonCount," << CommitReasonWithinHorizonCount << "\n";
    Out << "CommitReason_AheadOfHorizonCount," << CommitReasonAheadOfHorizonCount << "\n";
    Out << "CommitReason_EndOfStreamCount," << CommitReasonEndOfStreamCount << "\n";
    Out << "CommitReason_FallbackCount," << CommitReasonFallbackCount << "\n";
    Out << "CommitReason_UnknownCount," << CommitReasonUnknownCount << "\n";
    Out << "EvidenceMatchedEventCount," << TimingDiagnostics.MatchedEventCount << "\n";
    Out << "AnchorCoverageRate," << (TimingDiagnostics.MatchedEventCount > 0 ? static_cast<double>(TimingDiagnostics.EvidenceAnchoredEventCount) / TimingDiagnostics.MatchedEventCount : 0.0) << "\n";
    Out << "AnchoredMatchedEventCount," << TimingDiagnostics.AnchoredMatchedEventCount << "\n";
    Out << "RawTextTimingMAEMs," << (TimingDiagnostics.MatchedEventCount > 0 ? TimingDiagnostics.RawTextAbsErrorSumMs / TimingDiagnostics.MatchedEventCount : 0.0) << "\n";
    Out << "P90AbsRawTextTimingErrorMs," << Percentile90Ms(TimingDiagnostics.RawTextAbsErrorsMs) << "\n";
    Out << "P95AbsRawTextTimingErrorMs," << Percentile95Ms(TimingDiagnostics.RawTextAbsErrorsMs) << "\n";
    Out << "RawTextLeadBiasMs," << (TimingDiagnostics.MatchedEventCount > 0 ? TimingDiagnostics.RawTextLeadBiasSumMs / TimingDiagnostics.MatchedEventCount : 0.0) << "\n";
    Out << "StreamingTimingMAEMs," << (TimingDiagnostics.MatchedEventCount > 0 ? TimingDiagnostics.StreamingAbsErrorSumMs / TimingDiagnostics.MatchedEventCount : 0.0) << "\n";
    Out << "P90AbsStreamingTimingErrorMs," << Percentile90Ms(TimingDiagnostics.StreamingAbsErrorsMs) << "\n";
    Out << "P95AbsStreamingTimingErrorMs," << Percentile95Ms(TimingDiagnostics.StreamingAbsErrorsMs) << "\n";
    Out << "StreamingLeadBiasMs," << (TimingDiagnostics.MatchedEventCount > 0 ? TimingDiagnostics.StreamingLeadBiasSumMs / TimingDiagnostics.MatchedEventCount : 0.0) << "\n";
    Out << "AnchoredRawTextTimingMAEMs," << (TimingDiagnostics.AnchoredMatchedEventCount > 0 ? TimingDiagnostics.AnchoredRawTextAbsErrorSumMs / TimingDiagnostics.AnchoredMatchedEventCount : 0.0) << "\n";
    Out << "P90AbsAnchoredRawTextTimingErrorMs," << Percentile90Ms(TimingDiagnostics.AnchoredRawTextAbsErrorsMs) << "\n";
    Out << "P95AbsAnchoredRawTextTimingErrorMs," << Percentile95Ms(TimingDiagnostics.AnchoredRawTextAbsErrorsMs) << "\n";
    Out << "AnchoredStreamingTimingMAEMs," << (TimingDiagnostics.AnchoredMatchedEventCount > 0 ? TimingDiagnostics.AnchoredStreamingAbsErrorSumMs / TimingDiagnostics.AnchoredMatchedEventCount : 0.0) << "\n";
    Out << "P90AbsAnchoredStreamingTimingErrorMs," << Percentile90Ms(TimingDiagnostics.AnchoredStreamingAbsErrorsMs) << "\n";
    Out << "P95AbsAnchoredStreamingTimingErrorMs," << Percentile95Ms(TimingDiagnostics.AnchoredStreamingAbsErrorsMs) << "\n";
    Out << "MaxVisiblePoseGapSec," << PoseDiagnostics.MaxVisiblePoseGapSec << "\n";
    Out << "VisiblePoseGapCountOver250ms," << PoseDiagnostics.VisiblePoseGapCountOver250ms << "\n";
}

static FArgs ParseArgs(int argc, char** argv)
{
    FArgs A;
    for (int I = 1; I < argc; ++I)
    {
        const std::string K = argv[I];
        auto TakeValue = [&]() -> std::string
        {
            if (I + 1 >= argc) { return {}; }
            return argv[++I];
        };
        if (K == "--inputs") { A.InputsDir = TakeValue(); }
        else if (K == "--config") { A.ConfigPath = TakeValue(); }
        else if (K == "--out") { A.OutputDir = TakeValue(); }
        else if (K == "--chunk_ms") { A.ChunkMs = std::max(1, std::stoi(TakeValue())); A.bChunkMsOverridden = true; }
        else if (K == "--fps") { A.Fps = std::max(1, std::stoi(TakeValue())); A.bFpsOverridden = true; }
        else if (K == "--preroll_ms") { A.PrerollSec = std::max(0.0f, std::stof(TakeValue()) / 1000.0f); A.bPrerollOverridden = true; }
        else if (K == "--island_duration_scale") { A.IslandDurationScale = std::clamp(std::stof(TakeValue()), 0.75f, 1.35f); }
    }
    return A;
}

static bool ProcessCase(const fs::path& TranscriptPath, const fs::path& WavPath, const fs::path& CaseOutDir, const FArgs& Args, std::string& Error)
{
    const std::string CaseId = TranscriptPath.stem().string();
    const std::string Transcript = ReadTextFile(TranscriptPath);
    if (Transcript.empty())
    {
        Error = "empty transcript";
        return false;
    }

    FWavData Wav;
    if (!ReadWavPCM16(WavPath, Wav, Error))
    {
        return false;
    }
    if (Wav.SamplesInterleaved.empty())
    {
        Error = "empty WAV samples";
        return false;
    }
    if (Args.ExpectedSampleRateHz > 0 && Wav.SampleRate != Args.ExpectedSampleRateHz)
    {
        std::ostringstream SS;
        SS << "sample rate mismatch: expected " << Args.ExpectedSampleRateHz << " Hz, got " << Wav.SampleRate << " Hz";
        Error = SS.str();
        return false;
    }

    fs::create_directories(CaseOutDir);

    FOffgridAILipsyncRuntimeSession Runtime;
    FOffgridAILipsyncRuntimeBeginInput RuntimeInput;
    RuntimeInput.DialogueText = FString(Transcript.c_str());
    RuntimeInput.NPCID = FName("liplab");
    RuntimeInput.LineID = FName(CaseId.c_str());
    RuntimeInput.PrerollSec = Args.PrerollSec;
    RuntimeInput.IslandDurationScale = Args.IslandDurationScale;
    Runtime.BeginLine(RuntimeInput);


    const int FramesPerChunk = std::max(1, (Wav.SampleRate * Args.ChunkMs) / 1000);
    const int TotalFrames = static_cast<int>(Wav.SamplesInterleaved.size()) / std::max(1, Wav.Channels);

    for (int FrameOffset = 0; FrameOffset < TotalFrames; FrameOffset += FramesPerChunk)
    {
        const int FramesThisChunk = std::min(FramesPerChunk, TotalFrames - FrameOffset);
        TArray<uint8> ChunkBytes;
        ChunkBytes.Reserve(FramesThisChunk * Wav.Channels * 2);
        for (int F = 0; F < FramesThisChunk; ++F)
        {
            for (int C = 0; C < Wav.Channels; ++C)
            {
                const int16_t S = Wav.SamplesInterleaved[(FrameOffset + F) * Wav.Channels + C];
                ChunkBytes.Add(static_cast<uint8>(S & 0xff));
                ChunkBytes.Add(static_cast<uint8>((static_cast<uint16_t>(S) >> 8) & 0xff));
            }
        }
        Runtime.PushAudioPCM16(ChunkBytes, ChunkBytes.Num(), Wav.SampleRate, Wav.Channels, FrameOffset);
    }

    const float AudioDurationSec = static_cast<float>(TotalFrames) / static_cast<float>(std::max(1, Wav.SampleRate));
    const float FinalPlaybackSec = std::max(Runtime.GetPlaybackSeconds(), AudioDurationSec + 0.25f);
    Runtime.Finalize(FinalPlaybackSec);

    const FOffgridAITextVisemePlan& Plan = Runtime.GetTextPlan();
    const FOffgridAIStreamingSpeechDetector& Detector = Runtime.GetSpeechDetector();
    FOffgridAIAlignedVisemeTrack Track = Runtime.GetCommittedTrack();
    const FOffgridAIAlignedVisemeTrack PreNudgeTrack = Track;

    if (Args.bWriteDebugCsv)
    {
        WritePlannedEventsCsv(CaseOutDir / "planned_events.csv", Plan);
        WriteCommittedEventsCsv(CaseOutDir / "committed_events.csv", Track);
        WriteSpeechIslandsCsv(CaseOutDir / "speech_islands.csv", Detector.GetIslands());
        WriteAudioFeatureFramesCsv(CaseOutDir / "audio_feature_frames.csv", Detector.GetFeatureFrames());
        WriteFirstOnsetComparisonCsv(CaseOutDir / "first_onset_detector_comparison.csv", Detector.GetFeatureFrames(), Detector.GetIslands());
        WriteTextIslandDiagnosticsCsv(CaseOutDir / "text_island_diagnostics.csv", Transcript, Plan, Track);
        WriteRuntimeCommitEventsCsv(CaseOutDir / "runtime_commit_events.csv", Track);
        WriteRuntimeCommitEventsCsv(CaseOutDir / "streaming_effective_commit_events.csv", Track);
        WriteAudioOccupancyDiagnosticsCsv(CaseOutDir / "runtime_audio_occupancy_diagnostics.csv", Runtime.GetAudioOccupancyDiagnosticRows());
        WriteStreamTailDiagnosticsCsv(CaseOutDir / "runtime_stream_tail_diagnostics.csv", Runtime.GetStreamTailDiagnosticRow());
        WriteN10TimingCoverageDiagnostics(CaseOutDir, Track, Detector.GetIslands(), AudioDurationSec);
        WriteN11PacingOpportunityDiagnostics(CaseOutDir, Plan, Track, Detector.GetIslands(), Wav, Args.ChunkMs, AudioDurationSec, Args.PrerollSec);
        // AU41: speech fragments are detector-internal.  Keep the AU37 landmark
        // diagnostics, but do not write fragment-to-prosody assignment surfaces.
        WriteAU41NoFragmentRuntimeMetricsCsv(CaseOutDir / "au41_no_fragment_runtime_metrics.csv", Track, Detector.GetIslands(), Args.PrerollSec);
        // AU39: rollback unsafe runtime reflow. Keep detector/anchor diagnostics,
        // but measure the topology of the actual streaming-effective track.
        WriteAU39RuntimeTopologyMetricsCsv(CaseOutDir / "au39_runtime_topology_metrics.csv", Track, Args.PrerollSec);
        WriteDurationScalingDiagnosticsCsv(CaseOutDir / "duration_scaling_diagnostics.csv", CaseOutDir / "duration_scaling_summary.csv", Plan, Track, Detector.GetIslands());
    }
    const float ActiveSpeechStartSec = Detector.HasObservedFirstSpeechStart()
        ? Detector.GetFirstSpeechAudioBufferStartSec()
        : 0.0f;
    float ActiveSpeechEndSec = AudioDurationSec;
    if (Detector.GetIslands().Num() > 0)
    {
        ActiveSpeechEndSec = FMath::Clamp(
            GetSpeechIslandEndSec(Detector.GetIslands().Last(), AudioDurationSec),
            ActiveSpeechStartSec,
            AudioDurationSec);
    }
    const auto PreNudgePoseDiagnostics = CollectSubmittedPosePeaks(PreNudgeTrack, AudioDurationSec, Args.Fps, ActiveSpeechStartSec, ActiveSpeechEndSec, nullptr);

    // AU32: scored/effective track remains streaming-causal. Full-WAV audio is
    // used only to generate a landmark oracle for detector scoring; it must not
    // retime or mutate Track.

    const FTimingDiagnostics TimingDiagnostics = ComputeTimingDiagnostics(Plan, Wav, Track, Detector.GetIslands(), AudioDurationSec);
    const auto PoseDiagnostics = Args.bWriteDebugCsv
        ? WriteSubmittedPosesCsv(CaseOutDir / "submitted_poses.csv", Track, AudioDurationSec, Args.Fps, ActiveSpeechStartSec, ActiveSpeechEndSec)
        : CollectSubmittedPosePeaks(Track, AudioDurationSec, Args.Fps, ActiveSpeechStartSec, ActiveSpeechEndSec, nullptr);
    if (Args.bWriteDebugCsv)
    {
        WriteTimingDiagnosticsCsv(CaseOutDir / "timing_diagnostics.csv", TimingDiagnostics);
        WriteEventSubmissionSummaryCsv(CaseOutDir / "event_submission_summary.csv", Plan, Track, PoseDiagnostics);
        WriteNudgeDiagnosticsCsv(CaseOutDir / "nudge_diagnostics.csv", Track);
        WriteAlpha31NudgeOpportunityCsv(CaseOutDir / "alpha31_nudge_opportunities.csv", Plan, Wav, PreNudgeTrack, PreNudgePoseDiagnostics);
        WriteAu23VisemeAnchorFrontierCsv(CaseOutDir / "au23_viseme_anchor_frontier.csv", Plan, Wav, PreNudgeTrack);
        WriteAlpha32NudgeEffectivenessCsv(CaseOutDir / "alpha32_nudge_effectiveness.csv", PreNudgeTrack, PreNudgePoseDiagnostics, Track, PoseDiagnostics);
        WriteIslandLaunchDiagnosticsCsv(CaseOutDir / "island_launch_diagnostics.csv", Plan, Track, Detector.GetIslands());
        WriteRuntimeIslandAlignmentCsv(CaseOutDir / "runtime_island_alignment.csv", Plan, Track, Detector.GetIslands(), TimingDiagnostics, AudioDurationSec);
        WritePhraseDurationDiagnosticsCsv(CaseOutDir / "phrase_duration_diagnostics.csv", Transcript, Track, TimingDiagnostics);
        WriteSoftPunctuationBoundaryDiagnosticsCsv(CaseOutDir / "soft_punctuation_boundary_diagnostics.csv", Transcript, Track, Detector.GetIslands(), TimingDiagnostics, AudioDurationSec);
    }
    WriteScorecardCsv(CaseOutDir / "scorecard.csv", CaseId, Transcript, Plan, Track, PoseDiagnostics, TimingDiagnostics, Detector.GetIslands(), AudioDurationSec);
    return true;
}
} // namespace

int main(int argc, char** argv)
{
    FArgs Args = ParseArgs(argc, argv);
    std::string ConfigError;
    if (!LoadConfig(Args.ConfigPath, Args, ConfigError))
    {
        std::cerr << "CONFIG ERROR: " << ConfigError << "\n";
        return 2;
    }
    const fs::path TranscriptDir = Args.InputsDir / "transcripts";
    const fs::path WavDir = Args.InputsDir / "wav";

    std::cout << "liplab_runner\n";
    std::cout << "Inputs: " << Args.InputsDir.string() << "\n";
    std::cout << "Config: " << Args.ConfigPath.string() << "\n";
    std::cout << "Output: " << Args.OutputDir.string() << "\n";
    std::cout << "ChunkMs: " << Args.ChunkMs << "\n";
    std::cout << "PrerollMs: " << (Args.PrerollSec * 1000.0f) << "\n";
    std::cout << "IslandDurationScale: " << Args.IslandDurationScale << "\n";
    std::cout << "Fps: " << Args.Fps << "\n";
    std::cout << "WriteDebugCsv: " << (Args.bWriteDebugCsv ? "true" : "false") << "\n";
    if (Args.ExpectedSampleRateHz > 0)
    {
        std::cout << "ExpectedSampleRateHz: " << Args.ExpectedSampleRateHz << "\n";
    }

    if (!fs::exists(TranscriptDir) || !fs::exists(WavDir))
    {
        std::cerr << "Expected inputs layout:\n  " << TranscriptDir.string() << "\n  " << WavDir.string() << "\n";
        return 2;
    }

    fs::create_directories(Args.OutputDir / "per_case");
    std::vector<fs::path> TranscriptFiles;
    for (const auto& Entry : fs::directory_iterator(TranscriptDir))
    {
        if (Entry.is_regular_file() && Entry.path().extension() == ".txt")
        {
            TranscriptFiles.push_back(Entry.path());
        }
    }
    std::sort(TranscriptFiles.begin(), TranscriptFiles.end());

    int Ok = 0;
    int Failed = 0;
    std::ofstream Summary(Args.OutputDir / "run_status.csv");
    Summary << "CaseId,Status,Message\n";

    for (const fs::path& TranscriptPath : TranscriptFiles)
    {
        const std::string CaseId = TranscriptPath.stem().string();
        const fs::path WavPath = WavDir / (CaseId + ".wav");
        if (!fs::exists(WavPath))
        {
            ++Failed;
            Summary << CsvEscape(CaseId) << ",missing_wav," << CsvEscape(WavPath.string()) << "\n";
            std::cerr << "MISSING WAV: " << WavPath.string() << "\n";
            continue;
        }
        std::string Error;
        const fs::path CaseOutDir = Args.OutputDir / "per_case" / CaseId;
        if (ProcessCase(TranscriptPath, WavPath, CaseOutDir, Args, Error))
        {
            ++Ok;
            Summary << CsvEscape(CaseId) << ",ok,\n";
            std::cout << "OK " << CaseId << "\n";
        }
        else
        {
            ++Failed;
            Summary << CsvEscape(CaseId) << ",failed," << CsvEscape(Error) << "\n";
            std::cerr << "FAILED " << CaseId << ": " << Error << "\n";
        }
    }

    std::cout << "Processed: " << Ok << " ok, " << Failed << " failed\n";
    return Failed == 0 ? 0 : 1;
}
