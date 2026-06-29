#include "Lipsync/OffgridAIOnlinePhoneAligner.h"

namespace
{
static float Sat(float X)
{
    return FMath::Clamp(X, 0.0f, 1.0f);
}

static float Smooth01(float X, float Edge0, float Edge1)
{
    if (Edge1 <= Edge0) return X >= Edge1 ? 1.0f : 0.0f;
    const float T = Sat((X - Edge0) / (Edge1 - Edge0));
    return T * T * (3.0f - 2.0f * T);
}

static float SpeechScore(const FOffgridAIStreamingAudioFeatureFrame& F)
{
    return Sat(FMath::Max(F.RMSNorm, FMath::Max(F.Flux * 0.55f, F.Periodicity * 0.45f)));
}

static bool IsSpeechFrame(const FOffgridAIStreamingAudioFeatureFrame& F)
{
    return SpeechScore(F) >= 0.055f;
}

static float MaxDurationForClass(EOffgridAIPhoneClass C)
{
    switch (C)
    {
    case EOffgridAIPhoneClass::Bilabial: return 0.120f;
    case EOffgridAIPhoneClass::Labiodental: return 0.130f;
    case EOffgridAIPhoneClass::Dental: return 0.120f;
    case EOffgridAIPhoneClass::Sibilant: return 0.150f;
    case EOffgridAIPhoneClass::StopBurst: return 0.095f;
    case EOffgridAIPhoneClass::Liquid: return 0.170f;
    case EOffgridAIPhoneClass::Glide: return 0.140f;
    case EOffgridAIPhoneClass::Nasal: return 0.150f;
    case EOffgridAIPhoneClass::VowelOpen: return 0.220f;
    case EOffgridAIPhoneClass::VowelFront: return 0.210f;
    case EOffgridAIPhoneClass::VowelRound: return 0.230f;
    default: return 0.150f;
    }
}

static float MinDurationForClass(EOffgridAIPhoneClass C)
{
    switch (C)
    {
    case EOffgridAIPhoneClass::StopBurst: return 0.025f;
    case EOffgridAIPhoneClass::Bilabial: return 0.035f;
    case EOffgridAIPhoneClass::Sibilant: return 0.050f;
    case EOffgridAIPhoneClass::VowelOpen:
    case EOffgridAIPhoneClass::VowelFront:
    case EOffgridAIPhoneClass::VowelRound: return 0.060f;
    default: return 0.045f;
    }
}

static bool IsLikelyMatch(EOffgridAIPhoneClass C, float Score)
{
    if (C == EOffgridAIPhoneClass::Unknown) return Score >= 0.30f;
    if (C == EOffgridAIPhoneClass::VowelOpen || C == EOffgridAIPhoneClass::VowelFront || C == EOffgridAIPhoneClass::VowelRound) return Score >= 0.38f;
    if (C == EOffgridAIPhoneClass::StopBurst || C == EOffgridAIPhoneClass::Bilabial) return Score >= 0.34f;
    return Score >= 0.36f;
}
}

EOffgridAIPhoneClass FOffgridAIOnlinePhoneAligner::ClassForPhoneBase(const FString& PhoneBase)
{
    const FString P = PhoneBase.ToUpper();
    if (P.IsEmpty()) return EOffgridAIPhoneClass::Unknown;

    if (P == TEXT("AA") || P == TEXT("AE") || P == TEXT("AH") || P == TEXT("AW") || P == TEXT("AY")) return EOffgridAIPhoneClass::VowelOpen;
    if (P == TEXT("EH") || P == TEXT("ER") || P == TEXT("EY") || P == TEXT("IH") || P == TEXT("IY")) return EOffgridAIPhoneClass::VowelFront;
    if (P == TEXT("AO") || P == TEXT("OW") || P == TEXT("OY") || P == TEXT("UH") || P == TEXT("UW")) return EOffgridAIPhoneClass::VowelRound;

    if (P == TEXT("B") || P == TEXT("M") || P == TEXT("P")) return EOffgridAIPhoneClass::Bilabial;
    if (P == TEXT("F") || P == TEXT("V")) return EOffgridAIPhoneClass::Labiodental;
    if (P == TEXT("TH") || P == TEXT("DH")) return EOffgridAIPhoneClass::Dental;
    if (P == TEXT("S") || P == TEXT("Z") || P == TEXT("SH") || P == TEXT("ZH") || P == TEXT("CH") || P == TEXT("JH")) return EOffgridAIPhoneClass::Sibilant;
    if (P == TEXT("T") || P == TEXT("D") || P == TEXT("K") || P == TEXT("G") || P == TEXT("HH")) return EOffgridAIPhoneClass::StopBurst;
    if (P == TEXT("L") || P == TEXT("R")) return EOffgridAIPhoneClass::Liquid;
    if (P == TEXT("W") || P == TEXT("Y")) return EOffgridAIPhoneClass::Glide;
    if (P == TEXT("N") || P == TEXT("NG")) return EOffgridAIPhoneClass::Nasal;

    return EOffgridAIPhoneClass::Unknown;
}

FOffgridAIPhoneClassScores FOffgridAIOnlinePhoneAligner::ScoreFramePhoneClasses(const FOffgridAIStreamingAudioFeatureFrame& F)
{
    FOffgridAIPhoneClassScores S;
    const float Speech = SpeechScore(F);
    const float Voiced = Sat(F.Periodicity * 1.15f + F.RMSNorm * 0.35f - F.Flux * 0.18f);
    const float Frication = Sat(F.HighBandNorm * 0.65f + F.ZCR * 0.55f + F.SpectralCentroidNorm * 0.45f + F.Flux * 0.25f);
    const float Burst = Sat(F.Flux * 0.75f + (F.bLocalFluxPeak ? 0.30f : 0.0f) + F.DeltaRMS * 0.25f);
    const float LowCentroid = 1.0f - Sat(F.SpectralCentroidNorm);
    const float LowHighBand = 1.0f - Sat(F.HighBandNorm);
    const float Valley = (F.bLocalRMSValley ? 0.45f : 0.0f) + Smooth01(0.18f - F.RMSNorm, 0.0f, 0.18f) * 0.35f;

    S.Silence = Sat((1.0f - Speech) * 0.85f + (F.RMSNorm < 0.025f ? 0.25f : 0.0f));

    const float VowelBase = Sat(Voiced * 0.72f + F.RMSNorm * 0.45f + LowHighBand * 0.20f - F.Flux * 0.20f);
    S.VowelOpen = Sat(VowelBase * (0.65f + F.LowBandNorm * 0.30f + F.MidBandNorm * 0.20f));
    S.VowelFront = Sat(VowelBase * (0.52f + F.MidBandNorm * 0.32f + F.SpectralCentroidNorm * 0.22f));
    S.VowelRound = Sat(VowelBase * (0.56f + LowCentroid * 0.34f + F.LowBandNorm * 0.20f));

    S.Bilabial = Sat(Valley + Burst * 0.32f + LowCentroid * 0.18f);
    S.Labiodental = Sat(Frication * 0.55f + F.MidBandNorm * 0.25f + LowCentroid * 0.18f - F.HighBandNorm * 0.12f);
    S.Dental = Sat(Frication * 0.45f + F.MidBandNorm * 0.22f + F.ZCR * 0.20f);
    S.Sibilant = Sat(Frication * 0.78f + F.HighBandNorm * 0.32f + F.SpectralCentroidNorm * 0.30f);
    S.StopBurst = Sat(Burst * 0.78f + F.MidBandNorm * 0.18f + F.HighBandNorm * 0.12f);
    S.Liquid = Sat(Voiced * 0.52f + F.MidBandNorm * 0.22f + F.SpectralCentroidNorm * 0.16f);
    S.Glide = Sat(Voiced * 0.48f + LowCentroid * 0.18f + F.MidBandNorm * 0.16f);
    S.Nasal = Sat(Voiced * 0.56f + F.LowBandNorm * 0.32f + LowCentroid * 0.22f - F.HighBandNorm * 0.16f);
    S.Unknown = Sat(Speech * 0.35f + 0.05f);
    return S;
}

float FOffgridAIOnlinePhoneAligner::ScoreForClass(const FOffgridAIPhoneClassScores& Scores, EOffgridAIPhoneClass PhoneClass)
{
    switch (PhoneClass)
    {
    case EOffgridAIPhoneClass::Silence: return Scores.Silence;
    case EOffgridAIPhoneClass::VowelOpen: return Scores.VowelOpen;
    case EOffgridAIPhoneClass::VowelFront: return Scores.VowelFront;
    case EOffgridAIPhoneClass::VowelRound: return Scores.VowelRound;
    case EOffgridAIPhoneClass::Bilabial: return Scores.Bilabial;
    case EOffgridAIPhoneClass::Labiodental: return Scores.Labiodental;
    case EOffgridAIPhoneClass::Dental: return Scores.Dental;
    case EOffgridAIPhoneClass::Sibilant: return Scores.Sibilant;
    case EOffgridAIPhoneClass::StopBurst: return Scores.StopBurst;
    case EOffgridAIPhoneClass::Liquid: return Scores.Liquid;
    case EOffgridAIPhoneClass::Glide: return Scores.Glide;
    case EOffgridAIPhoneClass::Nasal: return Scores.Nasal;
    default: return Scores.Unknown;
    }
}

FString FOffgridAIOnlinePhoneAligner::PhoneClassToString(EOffgridAIPhoneClass PhoneClass)
{
    switch (PhoneClass)
    {
    case EOffgridAIPhoneClass::Silence: return TEXT("silence");
    case EOffgridAIPhoneClass::VowelOpen: return TEXT("vowel_open");
    case EOffgridAIPhoneClass::VowelFront: return TEXT("vowel_front");
    case EOffgridAIPhoneClass::VowelRound: return TEXT("vowel_round");
    case EOffgridAIPhoneClass::Bilabial: return TEXT("bilabial");
    case EOffgridAIPhoneClass::Labiodental: return TEXT("labiodental");
    case EOffgridAIPhoneClass::Dental: return TEXT("dental");
    case EOffgridAIPhoneClass::Sibilant: return TEXT("sibilant");
    case EOffgridAIPhoneClass::StopBurst: return TEXT("stop_burst");
    case EOffgridAIPhoneClass::Liquid: return TEXT("liquid");
    case EOffgridAIPhoneClass::Glide: return TEXT("glide");
    case EOffgridAIPhoneClass::Nasal: return TEXT("nasal");
    default: return TEXT("unknown");
    }
}

int32 FOffgridAIOnlinePhoneAligner::FindPhoneForEvent(const FOffgridAITextVisemePlan& Plan, const FOffgridAITextVisemeEvent& Event)
{
    int32 Fallback = INDEX_NONE;
    for (const FOffgridAIExpectedPhone& P : Plan.ExpectedPhones)
    {
        if (P.WordIndex == Event.WordIndex)
        {
            if (Fallback == INDEX_NONE) Fallback = P.PhoneIndex;
            if (Event.SourcePhoneIndex != INDEX_NONE && P.WordPhoneIndex == Event.SourcePhoneIndex)
            {
                return P.PhoneIndex;
            }
        }
    }
    return Fallback;
}

FOffgridAIOnlinePhoneAlignmentResult FOffgridAIOnlinePhoneAligner::Compute(const FOffgridAIOnlinePhoneAlignmentInput& Input)
{
    FOffgridAIOnlinePhoneAlignmentResult Out;
    if (!Input.Plan || Input.Plan->ExpectedPhones.Num() == 0)
    {
        return Out;
    }

    const int32 PhoneCount = Input.Plan->ExpectedPhones.Num();
    Out.PhoneStartSeconds.Init(-1.0f, PhoneCount);
    Out.PhoneCenterSeconds.Init(-1.0f, PhoneCount);
    Out.PhoneEndSeconds.Init(-1.0f, PhoneCount);
    Out.PhoneMatchScores.Init(0.0f, PhoneCount);
    Out.PhoneAdvanceReasons.Init(NAME_None, PhoneCount);

    if (!Input.AudioFeatureFrames || Input.AudioFeatureFrames->Num() == 0 || !Input.SpeechIslands || Input.SpeechIslands->Num() == 0)
    {
        return Out;
    }

    const float TimeLimit = Input.bFinal
        ? Input.ObservedAudioEndSec + 0.001f
        : FMath::Min(Input.ObservedAudioEndSec, Input.PlaybackSec + Input.LookaheadSec) + 0.001f;

    struct FAlignedFrameView
    {
        float Start = 0.0f;
        float End = 0.0f;
        float CumStart = 0.0f;
        float CumEnd = 0.0f;
        FOffgridAIPhoneClassScores Scores;
    };

    auto ConfirmedSpeechFrame = [&Input](const FOffgridAIStreamingAudioFeatureFrame& F) -> bool
    {
        const float Center = F.AudioBufferCenterSec;
        for (const FOffgridAIStreamingSpeechIsland& Island : *Input.SpeechIslands)
        {
            const float IslandStart = Island.AudioBufferStartSec - 0.001f;
            const float IslandEndSec = FMath::Max(Island.AudioBufferLastSpeechSec, Island.AudioBufferEndSec) + 0.001f;
            if (Center >= IslandStart && Center <= IslandEndSec)
            {
                return true;
            }
        }
        return false;
    };

    TArray<FAlignedFrameView> Frames;
    float ActiveSeconds = 0.0f;
    for (const FOffgridAIStreamingAudioFeatureFrame& F : *Input.AudioFeatureFrames)
    {
        if (F.AudioBufferEndSec > TimeLimit)
        {
            break;
        }
        if (!IsSpeechFrame(F) || !ConfirmedSpeechFrame(F))
        {
            continue;
        }

        const float FrameDuration = FMath::Max(F.AudioBufferEndSec - F.AudioBufferStartSec, 0.0f);
        if (FrameDuration <= 0.0f)
        {
            continue;
        }

        FAlignedFrameView V;
        V.Start = F.AudioBufferStartSec;
        V.End = F.AudioBufferEndSec;
        V.CumStart = ActiveSeconds;
        V.CumEnd = ActiveSeconds + FrameDuration;
        V.Scores = ScoreFramePhoneClasses(F);
        Frames.Add(V);
        ActiveSeconds += FrameDuration;
    }

    if (Frames.Num() == 0 || ActiveSeconds <= 0.001f)
    {
        return Out;
    }
    Out.bHasSpeechEvidence = true;

    auto TimeAtActive = [&Frames, ActiveSeconds](float ActiveSec) -> float
    {
        const float A = FMath::Clamp(ActiveSec, 0.0f, ActiveSeconds);
        for (const FAlignedFrameView& V : Frames)
        {
            if (A <= V.CumEnd + KINDA_SMALL_NUMBER)
            {
                const float Denom = FMath::Max(V.CumEnd - V.CumStart, 0.001f);
                const float Alpha = FMath::Clamp((A - V.CumStart) / Denom, 0.0f, 1.0f);
                return FMath::Lerp(V.Start, V.End, Alpha);
            }
        }
        return Frames.Last().End;
    };

    float ExpectedActiveSeconds = 0.0f;
    for (const FOffgridAIExpectedPhone& ExpectedPhone : Input.Plan->ExpectedPhones)
    {
        ExpectedActiveSeconds += FMath::Clamp(ExpectedPhone.WeightSeconds, 0.035f, 0.180f);
    }

    // Non-final streaming updates only align phones whose expected active duration
    // is already covered by observed speech frames. Once the TTS/audio input stream
    // has closed, all audio evidence is available even if playback has not reached it
    // yet; in that case, compress the expected phone-duration model into the observed
    // confirmed-speech mass so the transcript tail remains alignable to real audio
    // frames instead of being silently left for a fake final drain.
    const bool bCompletionPass = Input.bFinal && ExpectedActiveSeconds > ActiveSeconds + 0.001f;
    const float CompletionScale = bCompletionPass
        ? FMath::Clamp(ActiveSeconds / FMath::Max(ExpectedActiveSeconds, 0.001f), 0.05f, 1.0f)
        : 1.0f;

    float CumulativeActive = 0.0f;
    int32 LastAlignedPhone = INDEX_NONE;
    int32 LastAlignedWord = INDEX_NONE;

    for (int32 PhoneIndex = 0; PhoneIndex < PhoneCount; ++PhoneIndex)
    {
        const FOffgridAIExpectedPhone& Expected = Input.Plan->ExpectedPhones[PhoneIndex];
        const EOffgridAIPhoneClass ExpectedClass = ClassForPhoneBase(Expected.BasePhone);
        const float BaseWeight = FMath::Clamp(Expected.WeightSeconds, 0.035f, 0.180f);
        const float Weight = bCompletionPass
            ? FMath::Max(BaseWeight * CompletionScale, 0.0005f)
            : BaseWeight;
        const float StartActive = CumulativeActive;
        const float EndActive = StartActive + Weight;
        if (EndActive > ActiveSeconds + 0.001f)
        {
            if (!bCompletionPass || PhoneIndex == 0)
            {
                break;
            }
            // Use the last remaining confirmed-speech mass for the final phone when
            // small rounding differences would otherwise suppress the transcript tail.
        }
        const float ClampedEndActive = FMath::Min(EndActive, ActiveSeconds);
        CumulativeActive = ClampedEndActive;

        const float CenterActive = (StartActive + ClampedEndActive) * 0.5f;
        const float TargetCenter = TimeAtActive(CenterActive);
        const float StartTime = TimeAtActive(StartActive);
        const float EndTime = TimeAtActive(ClampedEndActive);

        float BestScore = 0.0f;
        float BestTime = TargetCenter;
        const float LocalSearchStart = FMath::Max(0.0f, StartActive - 0.060f);
        const float LocalSearchEnd = FMath::Min(ActiveSeconds, ClampedEndActive + 0.060f);
        for (const FAlignedFrameView& V : Frames)
        {
            if (V.CumEnd < LocalSearchStart || V.CumStart > LocalSearchEnd)
            {
                continue;
            }
            const float ClassScore = ScoreForClass(V.Scores, ExpectedClass);
            const float UnknownScore = ScoreForClass(V.Scores, EOffgridAIPhoneClass::Unknown);
            const float MatchScore = FMath::Max(ClassScore, UnknownScore * 0.45f);
            if (MatchScore > BestScore)
            {
                BestScore = MatchScore;
                BestTime = (V.Start + V.End) * 0.5f;
            }
        }

        const bool bClassSupported = IsLikelyMatch(ExpectedClass, BestScore);
        const float AcousticPull = bClassSupported ? 0.30f : 0.08f;
        const float CenterTime = FMath::Clamp(FMath::Lerp(TargetCenter, BestTime, AcousticPull), StartTime, EndTime);

        Out.PhoneStartSeconds[PhoneIndex] = StartTime;
        Out.PhoneEndSeconds[PhoneIndex] = EndTime;
        Out.PhoneCenterSeconds[PhoneIndex] = CenterTime;
        Out.PhoneMatchScores[PhoneIndex] = BestScore;
        Out.PhoneAdvanceReasons[PhoneIndex] = bCompletionPass
            ? FName(TEXT("frame_phone_confirmed_speech_completion"))
            : (bClassSupported
                ? FName(TEXT("frame_phone_confirmed_speech_phone_class"))
                : FName(TEXT("frame_phone_confirmed_speech_duration")));
        LastAlignedPhone = Expected.PhoneIndex;
        LastAlignedWord = Expected.WordIndex;
    }

    Out.HighestAlignedPhoneIndex = LastAlignedPhone;
    Out.HighestAlignedWordIndex = LastAlignedWord;
    if (Input.Plan->ExpectedPhones.IsValidIndex(LastAlignedPhone + 1))
    {
        Out.CurrentExpectedClass = ClassForPhoneBase(Input.Plan->ExpectedPhones[LastAlignedPhone + 1].BasePhone);
    }
    else if (Input.Plan->ExpectedPhones.IsValidIndex(LastAlignedPhone))
    {
        Out.CurrentExpectedClass = ClassForPhoneBase(Input.Plan->ExpectedPhones[LastAlignedPhone].BasePhone);
    }
    Out.CurrentExpectedClassScore = LastAlignedPhone != INDEX_NONE && Out.PhoneMatchScores.IsValidIndex(LastAlignedPhone)
        ? Out.PhoneMatchScores[LastAlignedPhone]
        : 0.0f;
    Out.AdvanceReason = bCompletionPass
        ? FName(TEXT("frame_phone_confirmed_speech_completion_alignment"))
        : FName(TEXT("frame_phone_confirmed_speech_global_alignment"));
    return Out;
}
