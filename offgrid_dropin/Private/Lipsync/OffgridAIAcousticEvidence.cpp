#include "Lipsync/OffgridAIAcousticEvidence.h"

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
}

EOffgridAIPhoneClass FOffgridAIAcousticEvidence::ClassForPhoneBase(const FString& PhoneBase)
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

FOffgridAIArticulatoryProbabilityField FOffgridAIAcousticEvidence::BuildArticulatoryProbabilityField(const FOffgridAIStreamingAudioFeatureFrame& F)
{
    FOffgridAIArticulatoryProbabilityField A;

    const float Speech = SpeechScore(F);
    const float Low = Sat(F.LowBandNorm);
    const float Mid = Sat(F.MidBandNorm);
    const float High = Sat(F.HighBandNorm);
    const float Centroid = Sat(F.SpectralCentroidNorm);
    const float LowCentroid = 1.0f - Centroid;
    const float HighCentroid = Centroid;
    const float LowHighTilt = Sat(Low - High + 0.42f);
    const float HighLowTilt = Sat(High - Low + 0.42f);

    const float PositiveEnergySlope = Sat(FMath::Max(F.DeltaRMS, 0.0f) * 1.65f);
    const float NegativeEnergySlope = Sat(FMath::Max(-F.DeltaRMS, 0.0f) * 1.65f);
    const float FluxPeak = Sat(F.Flux * 1.65f + (F.bLocalFluxPeak ? 0.28f : 0.0f));
    const float Valley = Sat((F.bLocalRMSValley ? 0.42f : 0.0f) + Smooth01(0.22f - F.RMSNorm, 0.0f, 0.22f) * 0.44f);

    A.Speech = Speech;
    A.Silence = Sat((1.0f - Speech) * 0.86f + (F.RMSNorm < 0.025f ? 0.25f : 0.0f));
    A.Voiced = Sat(F.Periodicity * 1.20f + F.RMSNorm * 0.30f - F.Flux * 0.16f - High * 0.08f);
    A.Fricative = Sat(((1.0f - F.Periodicity) * 0.48f + F.ZCR * 0.36f + High * 0.28f) * 0.64f + HighCentroid * 0.28f + F.Flux * 0.18f);
    A.Closure = Sat(Valley * 0.68f + NegativeEnergySlope * 0.18f + LowHighTilt * 0.16f - High * 0.10f);
    A.Release = Sat(FluxPeak * 0.70f + PositiveEnergySlope * 0.30f + High * 0.10f);
    A.Vowel = Sat(A.Voiced * 0.68f + F.RMSNorm * 0.36f - F.Flux * 0.24f - F.ZCR * 0.12f);
    A.Sonorant = Sat(A.Voiced * 0.56f + Low * 0.18f + Mid * 0.18f - A.Fricative * 0.12f - A.Closure * 0.08f);
    A.LowTilt = LowHighTilt;
    A.HighTilt = HighLowTilt;
    A.SpectralChange = Sat(F.Flux * 0.44f + F.ZCR * 0.22f + FMath::Abs(High - Low) * 0.16f + FMath::Abs(Centroid - 0.5f) * 0.10f);
    A.EnergyChange = Sat(PositiveEnergySlope * 0.55f + NegativeEnergySlope * 0.45f + (F.bLocalRMSValley ? 0.12f : 0.0f));

    // Transition probability is deliberately generic: it means "the articulatory
    // state is changing here", not "this exact frame is a word boundary".  The
    // runtime adapter aggregates this field over a window and combines it with
    // CMU-derived priors before it becomes a lexical-transition hint.
    A.Transition = Sat(0.30f * A.Closure + 0.26f * A.Release + 0.20f * A.SpectralChange + 0.14f * A.EnergyChange + 0.10f * FMath::Abs(A.Voiced - A.Fricative));

    const float SustainedVowel = Sat((F.RMSNorm - 0.18f) / 0.34f) * Sat((A.Vowel - 0.48f) / 0.36f) * Sat((0.38f - A.Transition) / 0.38f);
    const float FluxOnly = Sat(A.Release - 0.48f) * Sat((0.34f - A.Closure) / 0.34f) * Sat((0.34f - A.SpectralChange) / 0.34f);
    const float Flat = Sat((0.12f - A.EnergyChange) / 0.12f) * Sat((0.12f - A.SpectralChange) / 0.12f) * Sat((0.14f - A.Release) / 0.14f);
    A.RedHerring = Sat(0.48f * SustainedVowel + 0.34f * FluxOnly + 0.18f * Flat);

    FOffgridAIPhoneClassScores& S = A.PhoneScores;
    S.Silence = A.Silence;
    S.VowelOpen = Sat(A.Vowel * (0.48f + Low * 0.28f + Mid * 0.30f + F.RMSNorm * 0.12f - High * 0.08f));
    S.VowelFront = Sat(A.Vowel * (0.42f + Mid * 0.34f + Centroid * 0.26f + High * 0.10f - LowHighTilt * 0.08f));
    S.VowelRound = Sat(A.Vowel * (0.46f + LowCentroid * 0.32f + Low * 0.28f + LowHighTilt * 0.16f - High * 0.10f));
    S.Bilabial = Sat(A.Closure * 0.66f + A.Release * 0.18f + LowHighTilt * 0.16f + LowCentroid * 0.10f - A.Fricative * 0.18f);
    S.Labiodental = Sat(A.Fricative * 0.48f + Mid * 0.24f + LowHighTilt * 0.18f + LowCentroid * 0.10f - High * 0.10f);
    S.Dental = Sat(A.Fricative * 0.42f + Mid * 0.20f + F.ZCR * 0.22f + A.Release * 0.10f);
    S.Sibilant = Sat(A.Fricative * 0.78f + High * 0.34f + HighCentroid * 0.26f - Low * 0.10f);
    S.StopBurst = Sat(A.Release * 0.78f + Mid * 0.16f + High * 0.14f + A.Closure * 0.08f);
    S.Liquid = Sat(A.Voiced * 0.50f + Mid * 0.24f + Centroid * 0.14f + F.RMSNorm * 0.10f - A.Closure * 0.10f);
    S.Glide = Sat(A.Voiced * 0.48f + LowCentroid * 0.18f + Mid * 0.18f + F.RMSNorm * 0.08f - A.Release * 0.10f);
    S.Nasal = Sat(A.Voiced * 0.54f + Low * 0.34f + LowCentroid * 0.22f - High * 0.18f - A.Fricative * 0.08f);
    S.Unknown = Sat(Speech * 0.22f + 0.035f);
    return A;
}

FOffgridAIPhoneClassScores FOffgridAIAcousticEvidence::ScoreFramePhoneClasses(const FOffgridAIStreamingAudioFeatureFrame& F)
{
    return BuildArticulatoryProbabilityField(F).PhoneScores;
}

float FOffgridAIAcousticEvidence::ScoreForClass(const FOffgridAIPhoneClassScores& Scores, EOffgridAIPhoneClass PhoneClass)
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

FString FOffgridAIAcousticEvidence::PhoneClassToString(EOffgridAIPhoneClass PhoneClass)
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

