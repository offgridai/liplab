#include "OffgridAILineCoach.h"
#include "Lipsync/OffgridAILipsyncRuntimeAdapter.h"
#include "Lipsync/OffgridAIStreamingSpeechDetector.h"
#include "Lipsync/OffgridAITextVisemePlanner.h"

#include "HAL/FileManager.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"

namespace
{
static FString EscapeDebugCSVString(const FString& In)
{
    FString Out = In;
    Out.ReplaceInline(TEXT("\""), TEXT("\"\""));
    return FString::Printf(TEXT("\"%s\""), *Out);
}
}

namespace
{
struct FOffgridAIDebugTextLandmark
{
    int32 Index = INDEX_NONE;
    FString Kind;
    FString Family;
    int32 WordIndex = INDEX_NONE;
    FString Word;
    int32 PhoneIndex = INDEX_NONE;
    FString Phone;
    FString BasePhone;
    int32 SpeechRegionIndex = INDEX_NONE;
    int32 SentenceIndex = INDEX_NONE;
    float ExpectedSec = 0.0f;
    float StartSec = 0.0f;
    float EndSec = 0.0f;
    float Confidence = 0.0f;
    FString Reason;
};

struct FOffgridAIDebugPerceivedLandmark
{
    int32 Index = INDEX_NONE;
    FString Kind;
    FString Family;
    int32 FrameIndex = INDEX_NONE;
    int32 SpeechRegionIndex = INDEX_NONE;
    float CenterSec = 0.0f;
    float StartSec = 0.0f;
    float EndSec = 0.0f;
    float Confidence = 0.0f;
    float RMS = 0.0f;
    float RMSNorm = 0.0f;
    float SpeechEvidence = 0.0f;
    float Flux = 0.0f;
    float ZCR = 0.0f;
    float Periodicity = 0.0f;
    FString Reason;
};

struct FOffgridAIDebugBayesianLandmarkPrediction
{
    int32 FrameIndex = INDEX_NONE;
    float FrameSec = 0.0f;
    int32 TextLandmarkIndex = INDEX_NONE;
    FString Kind;
    FString Family;
    int32 WordIndex = INDEX_NONE;
    FString Word;
    FString Phone;
    float ExpectedSec = 0.0f;
    float HorizonSec = 0.0f;
    float Prior = 0.0f;
    float Likelihood = 0.0f;
    float Posterior = 0.0f;
    float EvidenceQuietRunSec = 0.0f;
    float EvidenceRMS = 0.0f;
    float EvidenceRMSNorm = 0.0f;
    float EvidenceSpeechEvidence = 0.0f;
    float EvidenceFlux = 0.0f;
    float EvidenceScore = 0.0f;
    FString EvidenceReason;
};

static int32 OffgridAIDebugStressLevel(const FString& Phone)
{
    if (Phone.Contains(TEXT("1"))) return 2;
    if (Phone.Contains(TEXT("2"))) return 1;
    return 0;
}

static bool OffgridAIDebugIsHardPause(TCHAR C, EOffgridAIBoundaryPauseClass PauseClass)
{
    return PauseClass == EOffgridAIBoundaryPauseClass::HardBreakPause
        || C == TEXT('.') || C == TEXT('!') || C == TEXT('?') || C == TEXT(':') || C == TEXT(';')
        || C == TEXT('-') || C == TEXT('—') || C == TEXT('–');
}

static FString OffgridAIDebugFamilyForBasePhone(const FString& Base, bool bIsVowel)
{
    if (bIsVowel) return TEXT("nucleus");
    if (Base == TEXT("M") || Base == TEXT("B") || Base == TEXT("P")) return TEXT("labial");
    if (Base == TEXT("F") || Base == TEXT("V") || Base == TEXT("S") || Base == TEXT("Z") || Base == TEXT("SH") || Base == TEXT("ZH") || Base == TEXT("TH") || Base == TEXT("DH")) return TEXT("fricative");
    if (Base == TEXT("T") || Base == TEXT("D") || Base == TEXT("K") || Base == TEXT("G") || Base == TEXT("CH") || Base == TEXT("JH")) return TEXT("burst");
    if (Base == TEXT("W") || Base == TEXT("Y") || Base == TEXT("R") || Base == TEXT("L")) return TEXT("glide_liquid");
    return TEXT("other_phone");
}

static float OffgridAIDebugPhoneLandmarkConfidence(const FOffgridAIExpectedPhone& Phone)
{
    const FString Base = Phone.BasePhone.ToUpper();
    if (Phone.bIsVowel)
    {
        const int32 Stress = OffgridAIDebugStressLevel(Phone.Phone);
        return Stress == 2 ? 1.0f : (Stress == 1 ? 0.80f : 0.62f);
    }
    if (Base == TEXT("M") || Base == TEXT("B") || Base == TEXT("P")) return 0.92f;
    if (Base == TEXT("F") || Base == TEXT("V") || Base == TEXT("S") || Base == TEXT("Z") || Base == TEXT("SH") || Base == TEXT("ZH") || Base == TEXT("TH") || Base == TEXT("DH")) return 0.82f;
    if (Base == TEXT("T") || Base == TEXT("D") || Base == TEXT("K") || Base == TEXT("G") || Base == TEXT("CH") || Base == TEXT("JH")) return 0.76f;
    return 0.0f;
}

static int32 OffgridAIDebugRegionIndexForTime(const TArray<FOffgridAIStreamingSpeechRegion>& Regions, float T)
{
    for (const FOffgridAIStreamingSpeechRegion& Region : Regions)
    {
        if (Region.bStarted && T >= Region.AudioBufferStartSec && T <= Region.AudioBufferEndSec + 0.040f)
        {
            return Region.SpeechRegionIndex;
        }
    }
    return INDEX_NONE;
}

static void OffgridAIDebugBuildPhonePriorTimes(const FOffgridAITextVisemePlan& Plan, float Duration, TArray<float>& OutStartSec, TArray<float>& OutCenterSec, TArray<float>& OutEndSec)
{
    float TotalWeight = 0.0f;
    for (const FOffgridAIExpectedPhone& Phone : Plan.ExpectedPhones)
    {
        TotalWeight += FMath::Max(Phone.WeightSeconds, 0.001f);
    }
    TotalWeight = FMath::Max(TotalWeight, 0.001f);

    OutStartSec.SetNumZeroed(Plan.ExpectedPhones.Num());
    OutCenterSec.SetNumZeroed(Plan.ExpectedPhones.Num());
    OutEndSec.SetNumZeroed(Plan.ExpectedPhones.Num());

    float AccumWeight = 0.0f;
    for (int32 PhoneIndex = 0; PhoneIndex < Plan.ExpectedPhones.Num(); ++PhoneIndex)
    {
        const float Weight = FMath::Max(Plan.ExpectedPhones[PhoneIndex].WeightSeconds, 0.001f);
        OutStartSec[PhoneIndex] = (AccumWeight / TotalWeight) * Duration;
        OutCenterSec[PhoneIndex] = ((AccumWeight + Weight * 0.5f) / TotalWeight) * Duration;
        AccumWeight += Weight;
        OutEndSec[PhoneIndex] = (AccumWeight / TotalWeight) * Duration;
    }
}

static float OffgridAIDebugLandmarkFrameScore(const FOffgridAIStreamingAudioFeatureFrame& F)
{
    return F.RMSNorm * (0.62f + 0.38f * F.SpeechEvidence) + 0.18f * F.Flux + 0.08f * F.Periodicity;
}

static bool OffgridAIDebugIsRuntimeQuiet(const FOffgridAIStreamingAudioFeatureFrame& F)
{
    // Quiet landmarks should be high precision.  A single low-evidence flag is
    // common inside consonants and must not create a lull by itself.
    const bool bDetectorQuiet = F.bStrongQuiet
        || (!F.bInSpeechAfterFrame && F.SpeechEvidence <= 0.34f)
        || (F.bLowEvidence && F.SpeechEvidence <= 0.24f);
    return F.RMS <= 0.0045f && bDetectorQuiet;
}

static void OffgridAIDebugBuildTextLandmarks(const FOffgridAITextVisemePlan& Plan, float Duration, TArray<FOffgridAIDebugTextLandmark>& Out)
{
    Out.Reset();
    TArray<float> PhoneStartSec;
    TArray<float> PhoneCenterSec;
    TArray<float> PhoneEndSec;
    OffgridAIDebugBuildPhonePriorTimes(Plan, Duration, PhoneStartSec, PhoneCenterSec, PhoneEndSec);

    auto AddTextLandmark = [&Out](FOffgridAIDebugTextLandmark L)
    {
        L.Index = Out.Num();
        Out.Add(L);
    };

    // Highest value text landmarks: punctuation-predicted lulls/silences.
    for (int32 WordIndex = 0; WordIndex < Plan.WordBoundaryPunctuationAfter.Num(); ++WordIndex)
    {
        const TCHAR Punc = Plan.WordBoundaryPunctuationAfter[WordIndex];
        if (Punc == TCHAR(0))
        {
            continue;
        }
        const EOffgridAIBoundaryPauseClass PauseClass = Plan.WordBoundaryPauseClassAfter.IsValidIndex(WordIndex)
            ? Plan.WordBoundaryPauseClassAfter[WordIndex]
            : EOffgridAIBoundaryPauseClass::None;
        const bool bHard = OffgridAIDebugIsHardPause(Punc, PauseClass);
        const int32 PhoneEndExclusive = Plan.WordPhoneEndIndices.IsValidIndex(WordIndex) ? Plan.WordPhoneEndIndices[WordIndex] : INDEX_NONE;
        const int32 LastPhoneIndex = PhoneEndExclusive > 0 ? PhoneEndExclusive - 1 : INDEX_NONE;
        float T = 0.0f;
        if (PhoneEndSec.IsValidIndex(LastPhoneIndex))
        {
            T = PhoneEndSec[LastPhoneIndex];
        }
        FOffgridAIDebugTextLandmark L;
        L.Kind = bHard ? TEXT("silence") : TEXT("prosody_lull");
        L.Family = TEXT("boundary");
        L.WordIndex = WordIndex;
        if (Plan.ExpectedPhones.IsValidIndex(LastPhoneIndex))
        {
            const FOffgridAIExpectedPhone& Phone = Plan.ExpectedPhones[LastPhoneIndex];
            L.Word = Phone.SourceWord;
            L.SpeechRegionIndex = Phone.SpeechRegionIndex;
            L.SentenceIndex = Phone.SentenceIndex;
        }
        L.PhoneIndex = LastPhoneIndex;
        L.Phone = Punc != TCHAR(0) ? FString::Chr(Punc) : FString();
        L.BasePhone = L.Phone;
        L.ExpectedSec = T;
        L.StartSec = FMath::Max(T - 0.060f, 0.0f);
        L.EndSec = T + (bHard ? 0.180f : 0.090f);
        L.Confidence = bHard ? 1.0f : 0.78f;
        L.Reason = bHard ? TEXT("text_hard_punctuation_silence") : TEXT("text_punctuation_prosody_lull");
        AddTextLandmark(L);
    }

    // Supporting text landmarks: vowel nuclei and strong visible consonant phones.
    for (int32 PhoneIndex = 0; PhoneIndex < Plan.ExpectedPhones.Num(); ++PhoneIndex)
    {
        const FOffgridAIExpectedPhone& Phone = Plan.ExpectedPhones[PhoneIndex];
        const FString Base = Phone.BasePhone.ToUpper();
        const FString Family = OffgridAIDebugFamilyForBasePhone(Base, Phone.bIsVowel);
        const float Confidence = OffgridAIDebugPhoneLandmarkConfidence(Phone);
        if (Confidence <= 0.0f || !PhoneCenterSec.IsValidIndex(PhoneIndex))
        {
            continue;
        }
        FOffgridAIDebugTextLandmark L;
        L.Kind = Phone.bIsVowel ? TEXT("nucleus_peak") : Family;
        L.Family = Family;
        L.WordIndex = Phone.WordIndex;
        L.Word = Phone.SourceWord;
        L.PhoneIndex = PhoneIndex;
        L.Phone = Phone.Phone;
        L.BasePhone = Base;
        L.SpeechRegionIndex = Phone.SpeechRegionIndex;
        L.SentenceIndex = Phone.SentenceIndex;
        L.ExpectedSec = PhoneCenterSec[PhoneIndex];
        L.StartSec = PhoneStartSec.IsValidIndex(PhoneIndex) ? PhoneStartSec[PhoneIndex] : L.ExpectedSec;
        L.EndSec = PhoneEndSec.IsValidIndex(PhoneIndex) ? PhoneEndSec[PhoneIndex] : L.ExpectedSec;
        L.Confidence = Confidence;
        L.Reason = Phone.bIsVowel ? TEXT("cmu_vowel_nucleus") : TEXT("cmu_strong_phone_landmark");
        AddTextLandmark(L);
    }

    Out.Sort([](const FOffgridAIDebugTextLandmark& A, const FOffgridAIDebugTextLandmark& B)
    {
        return A.ExpectedSec < B.ExpectedSec;
    });
    for (int32 I = 0; I < Out.Num(); ++I)
    {
        Out[I].Index = I;
    }
}

static void OffgridAIDebugBuildPerceivedLandmarks(const TArray<FOffgridAIStreamingAudioFeatureFrame>& Frames, const TArray<FOffgridAIStreamingSpeechRegion>& Regions, TArray<FOffgridAIDebugPerceivedLandmark>& Out)
{
    Out.Reset();
    auto AddPerceived = [&Out](FOffgridAIDebugPerceivedLandmark L)
    {
        L.Index = Out.Num();
        Out.Add(L);
    };

    // Long/medium quiet valleys: the most important runtime landmarks. Build a
    // conservative mask, then bridge a single noisy frame inside an otherwise
    // coherent valley. This improves recall without turning isolated consonant
    // closures into boundaries.
    TArray<bool> QuietMask;
    QuietMask.SetNumZeroed(Frames.Num());
    for (int32 I = 0; I < Frames.Num(); ++I)
    {
        QuietMask[I] = OffgridAIDebugIsRuntimeQuiet(Frames[I]);
    }
    for (int32 I = 1; I + 1 < Frames.Num(); ++I)
    {
        if (!QuietMask[I] && QuietMask[I - 1] && QuietMask[I + 1])
        {
            const FOffgridAIStreamingAudioFeatureFrame& F = Frames[I];
            const bool bNearQuiet = F.RMS <= 0.0055f && F.SpeechEvidence <= 0.40f;
            QuietMask[I] = bNearQuiet;
        }
    }

    int32 QuietRunStart = INDEX_NONE;
    for (int32 I = 0; I <= Frames.Num(); ++I)
    {
        const bool bQuiet = QuietMask.IsValidIndex(I) && QuietMask[I];
        if (bQuiet && QuietRunStart == INDEX_NONE)
        {
            QuietRunStart = I;
        }
        if ((!bQuiet || I == Frames.Num()) && QuietRunStart != INDEX_NONE)
        {
            const int32 QuietRunEnd = I - 1;
            if (Frames.IsValidIndex(QuietRunStart) && Frames.IsValidIndex(QuietRunEnd))
            {
                const float StartSec = Frames[QuietRunStart].AudioBufferStartSec;
                const float EndSec = Frames[QuietRunEnd].AudioBufferEndSec;
                const float Dur = EndSec - StartSec;
                if (Dur >= 0.030f)
                {
                    FOffgridAIDebugPerceivedLandmark L;
                    L.Kind = Dur >= 0.110f ? TEXT("silence") : TEXT("prosody_lull");
                    L.Family = TEXT("boundary");
                    L.FrameIndex = QuietRunStart;
                    L.SpeechRegionIndex = OffgridAIDebugRegionIndexForTime(Regions, StartSec);
                    L.StartSec = StartSec;
                    L.EndSec = EndSec;
                    float MinRawRMS = TNumericLimits<float>::Max();
                    float MinEvidence = 1.0f;
                    int32 ValleyFrameIndex = QuietRunStart;
                    int32 StrongQuietFrames = 0;
                    for (int32 Q = QuietRunStart; Q <= QuietRunEnd; ++Q)
                    {
                        if (Frames[Q].RMS < MinRawRMS)
                        {
                            MinRawRMS = Frames[Q].RMS;
                            ValleyFrameIndex = Q;
                        }
                        MinEvidence = FMath::Min(MinEvidence, Frames[Q].SpeechEvidence);
                        StrongQuietFrames += Frames[Q].bStrongQuiet ? 1 : 0;
                    }
                    // The minimum-energy frame is a more useful temporal landmark than
                    // the run midpoint, especially for asymmetric phrase decays.
                    L.CenterSec = Frames[ValleyFrameIndex].AudioBufferCenterSec;
                    L.FrameIndex = ValleyFrameIndex;
                    const float DurationConfidence = FMath::Clamp((Dur - 0.020f) / 0.160f, 0.0f, 1.0f);
                    const float DepthConfidence = FMath::Clamp((0.0050f - MinRawRMS) / 0.0050f, 0.0f, 1.0f);
                    const float DetectorConfidence = FMath::Clamp((0.34f - MinEvidence) / 0.34f, 0.0f, 1.0f);
                    const float StrongQuietRatio = float(StrongQuietFrames) / float(FMath::Max(QuietRunEnd - QuietRunStart + 1, 1));
                    L.Confidence = FMath::Clamp(0.45f * DurationConfidence + 0.30f * DepthConfidence + 0.15f * DetectorConfidence + 0.10f * StrongQuietRatio, 0.15f, 1.0f);
                    L.RMS = MinRawRMS;
                    L.RMSNorm = Frames[QuietRunStart].RMSNorm;
                    L.SpeechEvidence = MinEvidence;
                    L.Reason = Dur >= 0.110f ? TEXT("bridged_quiet_valley_silence") : TEXT("bridged_quiet_valley_lull");
                    AddPerceived(L);
                }
            }
            QuietRunStart = INDEX_NONE;
        }
    }

    constexpr int32 MinPeakFrameGap = 10;
    int32 LastAcceptedPeak = INDEX_NONE;
    int32 LastAcceptedBurst = INDEX_NONE;
    int32 LastAcceptedFricative = INDEX_NONE;
    int32 LastAcceptedLabial = INDEX_NONE;
    for (int32 I = 1; I + 1 < Frames.Num(); ++I)
    {
        const FOffgridAIStreamingAudioFeatureFrame& Prev = Frames[I - 1];
        const FOffgridAIStreamingAudioFeatureFrame& F = Frames[I];
        const FOffgridAIStreamingAudioFeatureFrame& Next = Frames[I + 1];
        const bool bInSpeech = F.bInSpeechAfterFrame || F.bInSpeechBeforeFrame;
        const float Score = OffgridAIDebugLandmarkFrameScore(F);
        const float PrevScore = OffgridAIDebugLandmarkFrameScore(Prev);
        const float NextScore = OffgridAIDebugLandmarkFrameScore(Next);

        const float LocalProminence = Score - 0.5f * (PrevScore + NextScore);
        if (bInSpeech
            && (F.bLocalRMSPeak || (Score >= PrevScore && Score > NextScore))
            && F.RMS >= 0.0045f
            && F.SpeechEvidence >= 0.18f
            && F.RMSNorm >= 0.18f
            && F.Periodicity >= 0.08f
            && Score >= 0.22f
            && LocalProminence >= 0.012f)
        {
            if (LastAcceptedPeak == INDEX_NONE || I - LastAcceptedPeak >= MinPeakFrameGap || Score > OffgridAIDebugLandmarkFrameScore(Frames[LastAcceptedPeak]))
            {
                if (LastAcceptedPeak != INDEX_NONE && I - LastAcceptedPeak < MinPeakFrameGap && Out.Num() > 0 && Out.Last().Kind == TEXT("nucleus_peak"))
                {
                    Out.Pop(EAllowShrinking::No);
                }
                FOffgridAIDebugPerceivedLandmark L;
                L.Kind = TEXT("nucleus_peak");
                L.Family = TEXT("nucleus");
                L.FrameIndex = I;
                L.SpeechRegionIndex = OffgridAIDebugRegionIndexForTime(Regions, F.AudioBufferCenterSec);
                L.CenterSec = F.AudioBufferCenterSec;
                L.StartSec = F.AudioBufferStartSec;
                L.EndSec = F.AudioBufferEndSec;
                L.Confidence = FMath::Clamp(Score, 0.0f, 1.0f);
                L.RMS = F.RMS;
                L.RMSNorm = F.RMSNorm;
                L.SpeechEvidence = F.SpeechEvidence;
                L.Flux = F.Flux;
                L.ZCR = F.ZCR;
                L.Periodicity = F.Periodicity;
                L.Reason = TEXT("local_energy_nucleus_peak");
                AddPerceived(L);
                LastAcceptedPeak = I;
            }
        }

        const bool bOnset = F.bStrongOnsetAnchor
            || (F.bLocalFluxPeak && F.Flux >= 0.155f && F.SpeechEvidence >= 0.16f)
            || (F.DeltaRMS >= 0.070f && Next.SpeechEvidence >= 0.17f);
        const bool bFric = F.HighBandNorm >= 0.14f && F.ZCR >= 0.15f && F.SpeechEvidence >= 0.14f && F.RMS >= 0.0025f;
        const bool bClosure = F.RMSNorm <= 0.12f && F.RMS <= 0.0045f && Next.Flux >= 0.145f && Next.SpeechEvidence >= 0.17f;
        const bool bBurstSpaced = LastAcceptedBurst == INDEX_NONE || I - LastAcceptedBurst >= 6;
        const bool bFricSpaced = LastAcceptedFricative == INDEX_NONE || I - LastAcceptedFricative >= 8;
        const bool bLabialSpaced = LastAcceptedLabial == INDEX_NONE || I - LastAcceptedLabial >= 8;
        if ((bOnset && bBurstSpaced) || (bFric && bFricSpaced) || (bClosure && bLabialSpaced))
        {
            FOffgridAIDebugPerceivedLandmark L;
            L.Kind = bFric ? TEXT("fricative") : (bClosure ? TEXT("labial") : TEXT("burst"));
            L.Family = L.Kind;
            L.FrameIndex = I;
            L.SpeechRegionIndex = OffgridAIDebugRegionIndexForTime(Regions, F.AudioBufferCenterSec);
            L.CenterSec = F.AudioBufferCenterSec;
            L.StartSec = F.AudioBufferStartSec;
            L.EndSec = F.AudioBufferEndSec;
            L.Confidence = FMath::Clamp((bOnset ? F.Flux * 0.65f : 0.0f) + (bFric ? F.HighBandNorm * 0.50f + F.ZCR * 0.25f : 0.0f) + F.SpeechEvidence * 0.25f, 0.0f, 1.0f);
            if (L.Confidence >= 0.18f)
            {
                L.RMS = F.RMS;
                L.RMSNorm = F.RMSNorm;
                L.SpeechEvidence = F.SpeechEvidence;
                L.Flux = F.Flux;
                L.ZCR = F.ZCR;
                L.Periodicity = F.Periodicity;
                L.Reason = bFric ? TEXT("high_band_zcr_fricative") : (bClosure ? TEXT("low_energy_pre_onset_closure") : TEXT("flux_or_delta_onset_burst"));
                AddPerceived(L);
                if (L.Kind == TEXT("burst")) LastAcceptedBurst = I;
                else if (L.Kind == TEXT("fricative")) LastAcceptedFricative = I;
                else if (L.Kind == TEXT("labial")) LastAcceptedLabial = I;
            }
        }
    }

    Out.Sort([](const FOffgridAIDebugPerceivedLandmark& A, const FOffgridAIDebugPerceivedLandmark& B)
    {
        return A.CenterSec < B.CenterSec;
    });
    for (int32 I = 0; I < Out.Num(); ++I)
    {
        Out[I].Index = I;
    }
}

static bool OffgridAIDebugLandmarkFamiliesCompatible(const FString& ExpectedFamily, const FString& PerceivedFamily)
{
    if (ExpectedFamily == PerceivedFamily) return true;
    if (ExpectedFamily == TEXT("boundary") && (PerceivedFamily == TEXT("boundary") || PerceivedFamily == TEXT("prosody_lull"))) return true;
    if (ExpectedFamily == TEXT("fricative") && PerceivedFamily == TEXT("fricative")) return true;
    if (ExpectedFamily == TEXT("labial") && (PerceivedFamily == TEXT("labial") || PerceivedFamily == TEXT("burst"))) return true;
    if (ExpectedFamily == TEXT("burst") && (PerceivedFamily == TEXT("burst") || PerceivedFamily == TEXT("labial"))) return true;
    return false;
}

static float OffgridAIDebugGaussianPrior(float DeltaSec, float SigmaSec)
{
    const float SafeSigma = FMath::Max(SigmaSec, 0.001f);
    const float Z = DeltaSec / SafeSigma;
    return FMath::Exp(-0.5f * Z * Z);
}

static float OffgridAIDebugBayesLikelihoodForLandmark(const FOffgridAIDebugTextLandmark& L, const FOffgridAIStreamingAudioFeatureFrame& F, float QuietRunSec, FString& OutReason, float& OutEvidenceScore)
{
    if (L.Family == TEXT("boundary"))
    {
        const float QuietScore = FMath::Clamp(QuietRunSec / 0.120f, 0.0f, 1.0f);
        const float EvidenceQuiet = FMath::Clamp((0.34f - F.SpeechEvidence) / 0.34f, 0.0f, 1.0f);
        const float RawQuiet = FMath::Clamp((0.0045f - F.RMS) / 0.0045f, 0.0f, 1.0f);
        OutEvidenceScore = FMath::Clamp(0.55f * QuietScore + 0.25f * RawQuiet + 0.20f * EvidenceQuiet, 0.0f, 1.0f);
        OutReason = QuietRunSec >= 0.110f ? TEXT("bayes_sustained_silence_evidence") : (QuietRunSec >= 0.030f ? TEXT("bayes_short_lull_evidence") : TEXT("bayes_no_lull_evidence"));
        return 0.08f + 0.86f * OutEvidenceScore;
    }

    if (L.Family == TEXT("nucleus"))
    {
        const float PeakScore = OffgridAIDebugLandmarkFrameScore(F);
        OutEvidenceScore = FMath::Clamp((PeakScore - 0.12f) / 0.55f, 0.0f, 1.0f);
        OutReason = OutEvidenceScore >= 0.60f ? TEXT("bayes_nucleus_energy_peak_evidence") : TEXT("bayes_weak_nucleus_evidence");
        return 0.12f + 0.78f * OutEvidenceScore;
    }

    if (L.Family == TEXT("fricative"))
    {
        OutEvidenceScore = FMath::Clamp(0.55f * F.HighBandNorm + 0.35f * F.ZCR + 0.25f * F.SpeechEvidence, 0.0f, 1.0f);
        OutReason = OutEvidenceScore >= 0.35f ? TEXT("bayes_high_band_fricative_evidence") : TEXT("bayes_weak_fricative_evidence");
        return 0.10f + 0.72f * OutEvidenceScore;
    }

    if (L.Family == TEXT("labial"))
    {
        const float ClosureScore = FMath::Clamp((0.18f - F.RMSNorm) / 0.18f, 0.0f, 1.0f);
        OutEvidenceScore = FMath::Clamp(0.45f * ClosureScore + 0.35f * F.Flux + 0.25f * F.SpeechEvidence, 0.0f, 1.0f);
        OutReason = OutEvidenceScore >= 0.35f ? TEXT("bayes_labial_closure_or_release_evidence") : TEXT("bayes_weak_labial_evidence");
        return 0.10f + 0.72f * OutEvidenceScore;
    }

    if (L.Family == TEXT("burst"))
    {
        OutEvidenceScore = FMath::Clamp(0.58f * F.Flux + 0.32f * F.DeltaRMS + 0.25f * F.SpeechEvidence, 0.0f, 1.0f);
        OutReason = OutEvidenceScore >= 0.32f ? TEXT("bayes_flux_burst_evidence") : TEXT("bayes_weak_burst_evidence");
        return 0.10f + 0.72f * OutEvidenceScore;
    }

    OutEvidenceScore = FMath::Clamp(F.SpeechEvidence, 0.0f, 1.0f);
    OutReason = TEXT("bayes_generic_landmark_evidence");
    return 0.10f + 0.55f * OutEvidenceScore;
}

static float OffgridAIDebugBayesPriorForLandmark(const FOffgridAIDebugTextLandmark& L, float HorizonSec)
{
    // Prior is intentionally text-and-time based only. Punctuation/lull landmarks get the strongest prior,
    // nuclei are useful secondary landmarks, and consonant landmarks are weak support hypotheses.
    const float AbsHorizon = FMath::Abs(HorizonSec);
    const float Sigma = L.Family == TEXT("boundary") ? 0.115f : (L.Family == TEXT("nucleus") ? 0.090f : 0.070f);
    const float TimePrior = OffgridAIDebugGaussianPrior(AbsHorizon, Sigma);
    const float FamilyPrior = L.Family == TEXT("boundary") ? 0.72f : (L.Family == TEXT("nucleus") ? 0.46f : 0.24f);
    return FMath::Clamp(0.02f + FamilyPrior * L.Confidence * TimePrior, 0.001f, 0.98f);
}

static void OffgridAIDebugBuildBayesianLandmarkPredictions(const TArray<FOffgridAIDebugTextLandmark>& TextLandmarks, const TArray<FOffgridAIStreamingAudioFeatureFrame>& Frames, TArray<FOffgridAIDebugBayesianLandmarkPrediction>& Out)
{
    Out.Reset();
    if (TextLandmarks.Num() == 0 || Frames.Num() == 0)
    {
        return;
    }

    TArray<bool> BayesQuietMask;
    BayesQuietMask.SetNumZeroed(Frames.Num());
    for (int32 I = 0; I < Frames.Num(); ++I)
    {
        BayesQuietMask[I] = OffgridAIDebugIsRuntimeQuiet(Frames[I]);
    }
    for (int32 I = 1; I + 1 < Frames.Num(); ++I)
    {
        if (!BayesQuietMask[I] && BayesQuietMask[I - 1] && BayesQuietMask[I + 1])
        {
            BayesQuietMask[I] = Frames[I].RMS <= 0.0055f && Frames[I].SpeechEvidence <= 0.40f;
        }
    }

    TArray<float> QuietRunSecByFrame;
    QuietRunSecByFrame.SetNumZeroed(Frames.Num());
    float QuietRunSec = 0.0f;
    for (int32 I = 0; I < Frames.Num(); ++I)
    {
        const bool bQuiet = BayesQuietMask[I];
        QuietRunSec = bQuiet ? (QuietRunSec + FMath::Max(Frames[I].AudioBufferEndSec - Frames[I].AudioBufferStartSec, 0.0f)) : 0.0f;
        QuietRunSecByFrame[I] = QuietRunSec;
    }

    constexpr int32 FrameStride = 2; // 20 ms at the current 10 ms feature cadence: compact but still streaming-like.
    constexpr int32 MaxCandidatesPerFrame = 6;
    constexpr float LookBehindSec = 0.060f;
    constexpr float LookAheadSec = 0.350f;

    TArray<float> RunningPosteriorByLandmark;
    RunningPosteriorByLandmark.Init(0.0f, TextLandmarks.Num());

    int32 LandmarkSearchStart = 0;
    for (int32 FrameIndex = 0; FrameIndex < Frames.Num(); FrameIndex += FrameStride)
    {
        const FOffgridAIStreamingAudioFeatureFrame& F = Frames[FrameIndex];
        const float T = F.AudioBufferCenterSec;
        while (TextLandmarks.IsValidIndex(LandmarkSearchStart) && TextLandmarks[LandmarkSearchStart].ExpectedSec < T - LookBehindSec)
        {
            ++LandmarkSearchStart;
        }

        TArray<FOffgridAIDebugBayesianLandmarkPrediction> Candidates;
        for (int32 LandmarkIndex = LandmarkSearchStart; LandmarkIndex < TextLandmarks.Num(); ++LandmarkIndex)
        {
            const FOffgridAIDebugTextLandmark& L = TextLandmarks[LandmarkIndex];
            const float HorizonSec = L.ExpectedSec - T;
            if (HorizonSec > LookAheadSec)
            {
                break;
            }
            if (HorizonSec < -LookBehindSec)
            {
                continue;
            }

            FOffgridAIDebugBayesianLandmarkPrediction P;
            P.FrameIndex = FrameIndex;
            P.FrameSec = T;
            P.TextLandmarkIndex = L.Index;
            P.Kind = L.Kind;
            P.Family = L.Family;
            P.WordIndex = L.WordIndex;
            P.Word = L.Word;
            P.Phone = L.Phone;
            P.ExpectedSec = L.ExpectedSec;
            P.HorizonSec = HorizonSec;
            P.Prior = OffgridAIDebugBayesPriorForLandmark(L, HorizonSec);
            P.Likelihood = OffgridAIDebugBayesLikelihoodForLandmark(L, F, QuietRunSecByFrame[FrameIndex], P.EvidenceReason, P.EvidenceScore);
            const float InstantPosterior = FMath::Clamp((P.Prior * P.Likelihood) / FMath::Max(P.Prior * P.Likelihood + (1.0f - P.Prior) * (1.0f - P.Likelihood), 0.0001f), 0.0f, 1.0f);
            const float PreviousPosterior = RunningPosteriorByLandmark.IsValidIndex(L.Index)
                ? RunningPosteriorByLandmark[L.Index]
                : 0.0f;
            const bool bSupportingEvidence = P.Likelihood >= 0.55f;
            float SequentialPosterior = bSupportingEvidence
                ? (1.0f - (1.0f - PreviousPosterior * 0.94f) * (1.0f - InstantPosterior * 0.58f))
                : FMath::Lerp(PreviousPosterior * 0.82f, InstantPosterior, 0.22f);
            if (HorizonSec < -0.020f)
            {
                SequentialPosterior *= 0.72f;
            }
            P.Posterior = FMath::Clamp(SequentialPosterior, 0.0f, 1.0f);
            if (RunningPosteriorByLandmark.IsValidIndex(L.Index))
            {
                RunningPosteriorByLandmark[L.Index] = P.Posterior;
            }
            P.EvidenceQuietRunSec = QuietRunSecByFrame[FrameIndex];
            P.EvidenceRMS = F.RMS;
            P.EvidenceRMSNorm = F.RMSNorm;
            P.EvidenceSpeechEvidence = F.SpeechEvidence;
            P.EvidenceFlux = F.Flux;
            Candidates.Add(P);
        }

        Candidates.Sort([](const FOffgridAIDebugBayesianLandmarkPrediction& A, const FOffgridAIDebugBayesianLandmarkPrediction& B)
        {
            if (A.Posterior != B.Posterior)
            {
                return A.Posterior > B.Posterior;
            }
            return FMath::Abs(A.HorizonSec) < FMath::Abs(B.HorizonSec);
        });

        const int32 NumToEmit = FMath::Min(MaxCandidatesPerFrame, Candidates.Num());
        for (int32 I = 0; I < NumToEmit; ++I)
        {
            if (Candidates[I].Posterior >= 0.06f || Candidates[I].Family == TEXT("boundary"))
            {
                Out.Add(Candidates[I]);
            }
        }
    }
}
}

void UOffgridAILineCoach::WriteLipsyncDebugLandmarksCSVs() const
{
    if (!bLipsyncDebugFileInitialized || LipsyncDebugLineDirectory.IsEmpty() || !LipsyncRuntimeSession)
    {
        return;
    }

    const FString LineString = bHasActiveLineRequest ? ActiveLineRequest.LineID.ToString() : FString(TEXT("None"));
    const FOffgridAITextVisemePlan& Plan = MirroredTextVisemePlan;
    const float Duration = FMath::Max(FMath::Max(MirroredTextPlanDurationSeconds, Plan.EstimatedDurationSeconds), 0.10f);

    TArray<FOffgridAIDebugTextLandmark> TextLandmarks;
    OffgridAIDebugBuildTextLandmarks(Plan, Duration, TextLandmarks);

    TArray<FOffgridAIDebugPerceivedLandmark> RuntimeLandmarks;
    OffgridAIDebugBuildPerceivedLandmarks(LipsyncRuntimeSession->GetAudioFeatureFrames(), LipsyncRuntimeSession->GetSpeechRegions(), RuntimeLandmarks);

    TArray<FOffgridAIDebugBayesianLandmarkPrediction> BayesianPredictions;
    OffgridAIDebugBuildBayesianLandmarkPredictions(TextLandmarks, LipsyncRuntimeSession->GetAudioFeatureFrames(), BayesianPredictions);

    FString TextCSV = TEXT("LineID,TextLandmarkIndex,Kind,Family,WordIndex,Word,PhoneIndex,Phone,BasePhone,SpeechRegionIndex,SentenceIndex,ExpectedSec,StartSec,EndSec,Confidence,Reason\n");
    for (const FOffgridAIDebugTextLandmark& L : TextLandmarks)
    {
        TArray<FString> C;
        C.Add(LineString);
        C.Add(FString::FromInt(L.Index));
        C.Add(EscapeDebugCSVString(L.Kind));
        C.Add(EscapeDebugCSVString(L.Family));
        C.Add(FString::FromInt(L.WordIndex));
        C.Add(EscapeDebugCSVString(L.Word));
        C.Add(L.PhoneIndex != INDEX_NONE ? FString::FromInt(L.PhoneIndex) : FString());
        C.Add(EscapeDebugCSVString(L.Phone));
        C.Add(EscapeDebugCSVString(L.BasePhone));
        C.Add(FString::FromInt(L.SpeechRegionIndex));
        C.Add(FString::FromInt(L.SentenceIndex));
        C.Add(FString::Printf(TEXT("%.6f"), L.ExpectedSec));
        C.Add(FString::Printf(TEXT("%.6f"), L.StartSec));
        C.Add(FString::Printf(TEXT("%.6f"), L.EndSec));
        C.Add(FString::Printf(TEXT("%.6f"), L.Confidence));
        C.Add(EscapeDebugCSVString(L.Reason));
        TextCSV += FString::Join(C, TEXT(",")) + TEXT("\n");
    }
    FFileHelper::SaveStringToFile(TextCSV, *FPaths::Combine(LipsyncDebugLineDirectory, TEXT("text_landmarks.csv")));

    FString RuntimeCSV = TEXT("LineID,RuntimeLandmarkIndex,Kind,Family,FrameIndex,SpeechRegionIndex,CenterSec,StartSec,EndSec,Confidence,RMS,RMSNorm,SpeechEvidence,Flux,ZCR,Periodicity,Reason\n");
    for (const FOffgridAIDebugPerceivedLandmark& L : RuntimeLandmarks)
    {
        TArray<FString> C;
        C.Add(LineString);
        C.Add(FString::FromInt(L.Index));
        C.Add(EscapeDebugCSVString(L.Kind));
        C.Add(EscapeDebugCSVString(L.Family));
        C.Add(FString::FromInt(L.FrameIndex));
        C.Add(FString::FromInt(L.SpeechRegionIndex));
        C.Add(FString::Printf(TEXT("%.6f"), L.CenterSec));
        C.Add(FString::Printf(TEXT("%.6f"), L.StartSec));
        C.Add(FString::Printf(TEXT("%.6f"), L.EndSec));
        C.Add(FString::Printf(TEXT("%.6f"), L.Confidence));
        C.Add(FString::Printf(TEXT("%.8f"), L.RMS));
        C.Add(FString::Printf(TEXT("%.6f"), L.RMSNorm));
        C.Add(FString::Printf(TEXT("%.6f"), L.SpeechEvidence));
        C.Add(FString::Printf(TEXT("%.6f"), L.Flux));
        C.Add(FString::Printf(TEXT("%.6f"), L.ZCR));
        C.Add(FString::Printf(TEXT("%.6f"), L.Periodicity));
        C.Add(EscapeDebugCSVString(L.Reason));
        RuntimeCSV += FString::Join(C, TEXT(",")) + TEXT("\n");
    }
    FFileHelper::SaveStringToFile(RuntimeCSV, *FPaths::Combine(LipsyncDebugLineDirectory, TEXT("runtime_landmarks.csv")));

    FString BayesCSV = TEXT("LineID,FrameIndex,FrameSec,TextLandmarkIndex,Kind,Family,WordIndex,Word,Phone,ExpectedSec,HorizonSec,Prior,Likelihood,Posterior,EvidenceQuietRunSec,EvidenceRMS,EvidenceRMSNorm,EvidenceSpeechEvidence,EvidenceFlux,EvidenceScore,EvidenceReason\n");
    for (const FOffgridAIDebugBayesianLandmarkPrediction& P : BayesianPredictions)
    {
        TArray<FString> C;
        C.Add(LineString);
        C.Add(FString::FromInt(P.FrameIndex));
        C.Add(FString::Printf(TEXT("%.6f"), P.FrameSec));
        C.Add(FString::FromInt(P.TextLandmarkIndex));
        C.Add(EscapeDebugCSVString(P.Kind));
        C.Add(EscapeDebugCSVString(P.Family));
        C.Add(FString::FromInt(P.WordIndex));
        C.Add(EscapeDebugCSVString(P.Word));
        C.Add(EscapeDebugCSVString(P.Phone));
        C.Add(FString::Printf(TEXT("%.6f"), P.ExpectedSec));
        C.Add(FString::Printf(TEXT("%.6f"), P.HorizonSec));
        C.Add(FString::Printf(TEXT("%.6f"), P.Prior));
        C.Add(FString::Printf(TEXT("%.6f"), P.Likelihood));
        C.Add(FString::Printf(TEXT("%.6f"), P.Posterior));
        C.Add(FString::Printf(TEXT("%.6f"), P.EvidenceQuietRunSec));
        C.Add(FString::Printf(TEXT("%.8f"), P.EvidenceRMS));
        C.Add(FString::Printf(TEXT("%.6f"), P.EvidenceRMSNorm));
        C.Add(FString::Printf(TEXT("%.6f"), P.EvidenceSpeechEvidence));
        C.Add(FString::Printf(TEXT("%.6f"), P.EvidenceFlux));
        C.Add(FString::Printf(TEXT("%.6f"), P.EvidenceScore));
        C.Add(EscapeDebugCSVString(P.EvidenceReason));
        BayesCSV += FString::Join(C, TEXT(",")) + TEXT("\n");
    }
    FFileHelper::SaveStringToFile(BayesCSV, *FPaths::Combine(LipsyncDebugLineDirectory, TEXT("landmark_bayes_predictions.csv")));

    FString MatchCSV = TEXT("LineID,TextLandmarkIndex,TextKind,TextFamily,WordIndex,Word,Phone,ExpectedSec,TextConfidence,MatchedRuntimeIndex,RuntimeKind,RuntimeFamily,RuntimeCenterSec,DeltaMs,RuntimeConfidence,MatchClass,MatchReason\n");
    TArray<int32> MatchedRuntimeIndexByText;
    TArray<float> MatchedRuntimeCenterSecByText;
    TArray<float> MatchedDeltaMsByText;
    TArray<float> MatchedRuntimeConfidenceByText;
    TArray<FString> MatchClassByText;
    TArray<FString> MatchReasonByText;
    MatchedRuntimeIndexByText.Init(INDEX_NONE, TextLandmarks.Num());
    MatchedRuntimeCenterSecByText.Init(-1.0f, TextLandmarks.Num());
    MatchedDeltaMsByText.Init(0.0f, TextLandmarks.Num());
    MatchedRuntimeConfidenceByText.Init(0.0f, TextLandmarks.Num());
    MatchClassByText.Init(TEXT("unmatched"), TextLandmarks.Num());
    MatchReasonByText.Init(TEXT("no_compatible_runtime_landmark"), TextLandmarks.Num());

    int32 RuntimeSearchStart = 0;
    TSet<int32> MatchedRuntime;
    for (const FOffgridAIDebugTextLandmark& E : TextLandmarks)
    {
        int32 BestIndex = INDEX_NONE;
        float BestCost = TNumericLimits<float>::Max();
        const float WindowBefore = E.Family == TEXT("boundary") ? 0.240f : 0.180f;
        const float WindowAfter = E.Family == TEXT("boundary") ? 0.420f : 0.260f;
        for (int32 RIndex = RuntimeSearchStart; RIndex < RuntimeLandmarks.Num(); ++RIndex)
        {
            const FOffgridAIDebugPerceivedLandmark& R = RuntimeLandmarks[RIndex];
            if (R.CenterSec < E.ExpectedSec - WindowBefore)
            {
                RuntimeSearchStart = FMath::Max(RuntimeSearchStart, RIndex);
                continue;
            }
            if (R.CenterSec > E.ExpectedSec + WindowAfter)
            {
                break;
            }
            if (!OffgridAIDebugLandmarkFamiliesCompatible(E.Family, R.Family))
            {
                continue;
            }
            const float Cost = FMath::Abs(R.CenterSec - E.ExpectedSec) - R.Confidence * 0.060f - E.Confidence * 0.030f;
            if (Cost < BestCost)
            {
                BestCost = Cost;
                BestIndex = RIndex;
            }
        }
        FString MatchClass = TEXT("unmatched");
        FString MatchReason = TEXT("no_compatible_runtime_landmark");
        float DeltaMs = 0.0f;
        if (BestIndex != INDEX_NONE)
        {
            const FOffgridAIDebugPerceivedLandmark& R = RuntimeLandmarks[BestIndex];
            MatchedRuntime.Add(BestIndex);
            RuntimeSearchStart = BestIndex + 1;
            DeltaMs = (R.CenterSec - E.ExpectedSec) * 1000.0f;
            const float AbsDelta = FMath::Abs(DeltaMs);
            MatchClass = AbsDelta <= 60.0f ? TEXT("tight") : (AbsDelta <= 140.0f ? TEXT("usable") : TEXT("loose"));
            MatchReason = TEXT("monotonic_nearest_compatible_landmark");

            if (MatchedRuntimeIndexByText.IsValidIndex(E.Index))
            {
                MatchedRuntimeIndexByText[E.Index] = BestIndex;
                MatchedRuntimeCenterSecByText[E.Index] = R.CenterSec;
                MatchedDeltaMsByText[E.Index] = DeltaMs;
                MatchedRuntimeConfidenceByText[E.Index] = R.Confidence;
                MatchClassByText[E.Index] = MatchClass;
                MatchReasonByText[E.Index] = MatchReason;
            }
        }
        else if (MatchClassByText.IsValidIndex(E.Index))
        {
            MatchClassByText[E.Index] = MatchClass;
            MatchReasonByText[E.Index] = MatchReason;
        }
        TArray<FString> C;
        C.Add(LineString);
        C.Add(FString::FromInt(E.Index));
        C.Add(EscapeDebugCSVString(E.Kind));
        C.Add(EscapeDebugCSVString(E.Family));
        C.Add(FString::FromInt(E.WordIndex));
        C.Add(EscapeDebugCSVString(E.Word));
        C.Add(EscapeDebugCSVString(E.Phone));
        C.Add(FString::Printf(TEXT("%.6f"), E.ExpectedSec));
        C.Add(FString::Printf(TEXT("%.6f"), E.Confidence));
        if (BestIndex != INDEX_NONE)
        {
            const FOffgridAIDebugPerceivedLandmark& R = RuntimeLandmarks[BestIndex];
            C.Add(FString::FromInt(R.Index));
            C.Add(EscapeDebugCSVString(R.Kind));
            C.Add(EscapeDebugCSVString(R.Family));
            C.Add(FString::Printf(TEXT("%.6f"), R.CenterSec));
            C.Add(FString::Printf(TEXT("%.3f"), DeltaMs));
            C.Add(FString::Printf(TEXT("%.6f"), R.Confidence));
        }
        else
        {
            C.Add(TEXT("")); C.Add(TEXT("")); C.Add(TEXT("")); C.Add(TEXT("")); C.Add(TEXT("")); C.Add(TEXT(""));
        }
        C.Add(EscapeDebugCSVString(MatchClass));
        C.Add(EscapeDebugCSVString(MatchReason));
        MatchCSV += FString::Join(C, TEXT(",")) + TEXT("\n");
    }

    for (int32 RIndex = 0; RIndex < RuntimeLandmarks.Num(); ++RIndex)
    {
        if (MatchedRuntime.Contains(RIndex))
        {
            continue;
        }
        const FOffgridAIDebugPerceivedLandmark& R = RuntimeLandmarks[RIndex];
        TArray<FString> C;
        C.Add(LineString);
        C.Add(TEXT("")); C.Add(TEXT("")); C.Add(TEXT("")); C.Add(TEXT("")); C.Add(TEXT("")); C.Add(TEXT("")); C.Add(TEXT("")); C.Add(TEXT(""));
        C.Add(FString::FromInt(R.Index));
        C.Add(EscapeDebugCSVString(R.Kind));
        C.Add(EscapeDebugCSVString(R.Family));
        C.Add(FString::Printf(TEXT("%.6f"), R.CenterSec));
        C.Add(TEXT(""));
        C.Add(FString::Printf(TEXT("%.6f"), R.Confidence));
        C.Add(TEXT("unmatched_runtime_landmark"));
        C.Add(EscapeDebugCSVString(R.Reason));
        MatchCSV += FString::Join(C, TEXT(",")) + TEXT("\n");
    }

    FFileHelper::SaveStringToFile(MatchCSV, *FPaths::Combine(LipsyncDebugLineDirectory, TEXT("landmark_matches.csv")));

    FString BayesScoreCSV = TEXT("LineID,TextLandmarkIndex,Kind,Family,WordIndex,Word,Phone,ExpectedSec,TextConfidence,MatchedRuntimeIndex,RuntimeCenterSec,RuntimeDeltaMs,RuntimeConfidence,MatchClass,EvalSec,BestPosteriorPreEval,BestPredictionFrameSec,BestPredictionLeadMs,BestPriorPreEval,BestLikelihoodPreEval,BestEvidenceReason,PosteriorNearExpected,PosteriorNearRuntime,FirstPosterior50Sec,FirstPosterior50LeadMs,FirstPosterior75Sec,FirstPosterior75LeadMs,PredictionClass,PredictionReason\n");
    for (const FOffgridAIDebugTextLandmark& E : TextLandmarks)
    {
        const int32 MatchedRuntimeIndex = MatchedRuntimeIndexByText.IsValidIndex(E.Index) ? MatchedRuntimeIndexByText[E.Index] : INDEX_NONE;
        const float RuntimeCenterSec = MatchedRuntimeCenterSecByText.IsValidIndex(E.Index) ? MatchedRuntimeCenterSecByText[E.Index] : -1.0f;
        const float RuntimeDeltaMs = MatchedDeltaMsByText.IsValidIndex(E.Index) ? MatchedDeltaMsByText[E.Index] : 0.0f;
        const float RuntimeConfidence = MatchedRuntimeConfidenceByText.IsValidIndex(E.Index) ? MatchedRuntimeConfidenceByText[E.Index] : 0.0f;
        const FString MatchClass = MatchClassByText.IsValidIndex(E.Index) ? MatchClassByText[E.Index] : FString(TEXT("unmatched"));
        const FString MatchReason = MatchReasonByText.IsValidIndex(E.Index) ? MatchReasonByText[E.Index] : FString(TEXT("no_compatible_runtime_landmark"));
        const float EvalSec = RuntimeCenterSec >= 0.0f ? RuntimeCenterSec : E.ExpectedSec;

        float BestPosterior = -1.0f;
        float BestPrior = 0.0f;
        float BestLikelihood = 0.0f;
        float BestFrameSec = -1.0f;
        FString BestEvidenceReason;

        float PosteriorNearExpected = -1.0f;
        float PosteriorNearExpectedAbsDt = TNumericLimits<float>::Max();
        float PosteriorNearRuntime = -1.0f;
        float PosteriorNearRuntimeAbsDt = TNumericLimits<float>::Max();

        float FirstPosterior50Sec = -1.0f;
        float FirstPosterior75Sec = -1.0f;

        for (const FOffgridAIDebugBayesianLandmarkPrediction& P : BayesianPredictions)
        {
            if (P.TextLandmarkIndex != E.Index)
            {
                continue;
            }

            const float ExpectedAbsDt = FMath::Abs(P.FrameSec - E.ExpectedSec);
            if (ExpectedAbsDt < PosteriorNearExpectedAbsDt)
            {
                PosteriorNearExpectedAbsDt = ExpectedAbsDt;
                PosteriorNearExpected = P.Posterior;
            }

            if (RuntimeCenterSec >= 0.0f)
            {
                const float RuntimeAbsDt = FMath::Abs(P.FrameSec - RuntimeCenterSec);
                if (RuntimeAbsDt < PosteriorNearRuntimeAbsDt)
                {
                    PosteriorNearRuntimeAbsDt = RuntimeAbsDt;
                    PosteriorNearRuntime = P.Posterior;
                }
            }

            if (P.FrameSec <= EvalSec + 0.001f)
            {
                if (P.Posterior > BestPosterior)
                {
                    BestPosterior = P.Posterior;
                    BestPrior = P.Prior;
                    BestLikelihood = P.Likelihood;
                    BestFrameSec = P.FrameSec;
                    BestEvidenceReason = P.EvidenceReason;
                }
                if (FirstPosterior50Sec < 0.0f && P.Posterior >= 0.50f)
                {
                    FirstPosterior50Sec = P.FrameSec;
                }
                if (FirstPosterior75Sec < 0.0f && P.Posterior >= 0.75f)
                {
                    FirstPosterior75Sec = P.FrameSec;
                }
            }
        }

        const float BestLeadMs = BestFrameSec >= 0.0f ? (EvalSec - BestFrameSec) * 1000.0f : 0.0f;
        const float First50LeadMs = FirstPosterior50Sec >= 0.0f ? (EvalSec - FirstPosterior50Sec) * 1000.0f : 0.0f;
        const float First75LeadMs = FirstPosterior75Sec >= 0.0f ? (EvalSec - FirstPosterior75Sec) * 1000.0f : 0.0f;

        FString PredictionClass;
        FString PredictionReason;
        if (MatchedRuntimeIndex == INDEX_NONE)
        {
            PredictionClass = TEXT("no_runtime_match");
            PredictionReason = MatchReason;
        }
        else if (BestPosterior >= 0.75f && FirstPosterior75Sec >= 0.0f && First75LeadMs >= 40.0f)
        {
            PredictionClass = TEXT("strong_predictive");
            PredictionReason = TEXT("posterior_75_before_runtime_match");
        }
        else if (BestPosterior >= 0.50f && FirstPosterior50Sec >= 0.0f && First50LeadMs >= 20.0f)
        {
            PredictionClass = TEXT("predictive");
            PredictionReason = TEXT("posterior_50_before_runtime_match");
        }
        else if (PosteriorNearRuntime >= 0.50f)
        {
            PredictionClass = TEXT("detected_at_match");
            PredictionReason = TEXT("posterior_high_near_runtime_match_but_not_early");
        }
        else if (BestPosterior >= 0.20f)
        {
            PredictionClass = TEXT("weak_hint");
            PredictionReason = TEXT("posterior_nontrivial_before_match");
        }
        else
        {
            PredictionClass = TEXT("not_predicted");
            PredictionReason = TEXT("posterior_low_before_match");
        }

        TArray<FString> C;
        C.Add(LineString);
        C.Add(FString::FromInt(E.Index));
        C.Add(EscapeDebugCSVString(E.Kind));
        C.Add(EscapeDebugCSVString(E.Family));
        C.Add(FString::FromInt(E.WordIndex));
        C.Add(EscapeDebugCSVString(E.Word));
        C.Add(EscapeDebugCSVString(E.Phone));
        C.Add(FString::Printf(TEXT("%.6f"), E.ExpectedSec));
        C.Add(FString::Printf(TEXT("%.6f"), E.Confidence));
        C.Add(MatchedRuntimeIndex != INDEX_NONE ? FString::FromInt(MatchedRuntimeIndex) : FString());
        C.Add(RuntimeCenterSec >= 0.0f ? FString::Printf(TEXT("%.6f"), RuntimeCenterSec) : FString());
        C.Add(MatchedRuntimeIndex != INDEX_NONE ? FString::Printf(TEXT("%.3f"), RuntimeDeltaMs) : FString());
        C.Add(MatchedRuntimeIndex != INDEX_NONE ? FString::Printf(TEXT("%.6f"), RuntimeConfidence) : FString());
        C.Add(EscapeDebugCSVString(MatchClass));
        C.Add(FString::Printf(TEXT("%.6f"), EvalSec));
        C.Add(BestPosterior >= 0.0f ? FString::Printf(TEXT("%.6f"), BestPosterior) : FString());
        C.Add(BestFrameSec >= 0.0f ? FString::Printf(TEXT("%.6f"), BestFrameSec) : FString());
        C.Add(BestFrameSec >= 0.0f ? FString::Printf(TEXT("%.3f"), BestLeadMs) : FString());
        C.Add(BestPosterior >= 0.0f ? FString::Printf(TEXT("%.6f"), BestPrior) : FString());
        C.Add(BestPosterior >= 0.0f ? FString::Printf(TEXT("%.6f"), BestLikelihood) : FString());
        C.Add(EscapeDebugCSVString(BestEvidenceReason));
        C.Add(PosteriorNearExpected >= 0.0f ? FString::Printf(TEXT("%.6f"), PosteriorNearExpected) : FString());
        C.Add(PosteriorNearRuntime >= 0.0f ? FString::Printf(TEXT("%.6f"), PosteriorNearRuntime) : FString());
        C.Add(FirstPosterior50Sec >= 0.0f ? FString::Printf(TEXT("%.6f"), FirstPosterior50Sec) : FString());
        C.Add(FirstPosterior50Sec >= 0.0f ? FString::Printf(TEXT("%.3f"), First50LeadMs) : FString());
        C.Add(FirstPosterior75Sec >= 0.0f ? FString::Printf(TEXT("%.6f"), FirstPosterior75Sec) : FString());
        C.Add(FirstPosterior75Sec >= 0.0f ? FString::Printf(TEXT("%.3f"), First75LeadMs) : FString());
        C.Add(EscapeDebugCSVString(PredictionClass));
        C.Add(EscapeDebugCSVString(PredictionReason));
        BayesScoreCSV += FString::Join(C, TEXT(",")) + TEXT("\n");
    }
    FFileHelper::SaveStringToFile(BayesScoreCSV, *FPaths::Combine(LipsyncDebugLineDirectory, TEXT("landmark_bayes_scores.csv")));
}


