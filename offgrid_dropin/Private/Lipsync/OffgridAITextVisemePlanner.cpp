#include "Lipsync/OffgridAITextVisemePlanner.h"
#include "Lipsync/OffgridAICmudictData.h"

namespace
{
#include "OffgridAITtsPronunciationPreferences.inl"

static FString NormalizeWord(const FString& In)
{
    FString Out;
    for (TCHAR C : In)
    {
        if (FChar::IsAlnum(C) || C == TCHAR('\''))
        {
            Out.AppendChar(FChar::ToLower(C));
        }
    }
    return Out;
}

static bool IsHardSentenceBoundary(TCHAR C)
{
    return C == TEXT('.')
        || C == TEXT('!')
        || C == TEXT('?')
        || C == TEXT(':')
        || C == TEXT(';')
        || C == TEXT('-')
        || C == TEXT('—')
        || C == TEXT('–');
}

static bool IsSpeechRegionBoundary(TCHAR C)
{
    return IsHardSentenceBoundary(C) || C == TEXT(',');
}

static bool IsAsciiHyphenPunctuationAt(const FString& Text, int32 Index)
{
    if (Index < 0 || Index >= Text.Len() || Text[Index] != TEXT('-'))
    {
        return false;
    }

    // A plain ASCII hyphen is lexical inside compounds such as
    // "kettle-cooked" and should not create a speech-region boundary there.
    // Treat it as dash punctuation only when it is visually separated as its
    // own token: "word - word". En/em dashes remain punctuation elsewhere.
    const bool bHasLeft = Index > 0;
    const bool bHasRight = Index + 1 < Text.Len();
    const TCHAR Left = bHasLeft ? Text[Index - 1] : TCHAR(0);
    const TCHAR Right = bHasRight ? Text[Index + 1] : TCHAR(0);
    return bHasLeft && bHasRight && FChar::IsWhitespace(Left) && FChar::IsWhitespace(Right);
}

static bool IsSpeechRegionBoundaryAt(const FString& Text, int32 Index)
{
    if (Index < 0 || Index >= Text.Len())
    {
        return false;
    }

    const TCHAR C = Text[Index];
    if (C == TEXT('-'))
    {
        return IsAsciiHyphenPunctuationAt(Text, Index);
    }
    return IsSpeechRegionBoundary(C);
}

static bool IsWordTokenCharAt(const FString& Text, int32 Index)
{
    if (Index < 0 || Index >= Text.Len())
    {
        return false;
    }

    const TCHAR C = Text[Index];
    return FChar::IsAlnum(C) || C == TCHAR('\'');
}

static bool IsListConjunctionWord(const FString& Word)
{
    return Word == TEXT("and") || Word == TEXT("or");
}

static bool IsClauseStarterWord(const FString& Word)
{
    static const TSet<FString> ClauseStarters = {
        TEXT("what"), TEXT("when"), TEXT("where"), TEXT("why"), TEXT("how"),
        TEXT("who"), TEXT("whom"), TEXT("whose"), TEXT("which"),
        TEXT("can"), TEXT("could"), TEXT("would"), TEXT("will"), TEXT("did"),
        TEXT("do"), TEXT("does"), TEXT("is"), TEXT("are"), TEXT("was"), TEXT("were"),
        TEXT("have"), TEXT("has"), TEXT("had"),
        TEXT("i"), TEXT("you"), TEXT("we"), TEXT("they"), TEXT("he"), TEXT("she"), TEXT("it"),
        TEXT("though"), TEXT("because"), TEXT("but"), TEXT("so"), TEXT("then"), TEXT("well")
    };
    return ClauseStarters.Contains(Word);
}

static EOffgridAIBoundaryPauseClass ClassifyBoundaryPause(
    const TArray<FString>& Words,
    const TArray<TCHAR>& Boundaries,
    int32 WordIndex)
{
    const TCHAR Boundary = Boundaries.IsValidIndex(WordIndex) ? Boundaries[WordIndex] : TCHAR(0);
    if (Boundary == TCHAR(0))
    {
        return EOffgridAIBoundaryPauseClass::None;
    }

    if (IsHardSentenceBoundary(Boundary))
    {
        return EOffgridAIBoundaryPauseClass::HardBreakPause;
    }

    if (Boundary != TEXT(','))
    {
        return EOffgridAIBoundaryPauseClass::None;
    }

    int32 SentenceStart = 0;
    for (int32 I = WordIndex - 1; I >= 0; --I)
    {
        if (Boundaries.IsValidIndex(I) && IsHardSentenceBoundary(Boundaries[I]))
        {
            SentenceStart = I + 1;
            break;
        }
    }

    int32 SentenceEnd = Words.Num() - 1;
    for (int32 I = WordIndex + 1; I < Boundaries.Num(); ++I)
    {
        if (IsHardSentenceBoundary(Boundaries[I]))
        {
            SentenceEnd = I;
            break;
        }
    }

    int32 CommaCountInSentence = 0;
    bool bHasListConjunctionAhead = false;
    for (int32 I = SentenceStart; I <= SentenceEnd && I < Boundaries.Num(); ++I)
    {
        if (Boundaries[I] == TEXT(','))
        {
            ++CommaCountInSentence;
        }
        if (I > WordIndex && I < Words.Num() && IsListConjunctionWord(Words[I]))
        {
            bHasListConjunctionAhead = true;
        }
    }

    const FString NextWord = Words.IsValidIndex(WordIndex + 1) ? Words[WordIndex + 1] : FString();
    const bool bClauseStarterAhead = !NextWord.IsEmpty() && IsClauseStarterWord(NextWord);
    const bool bLooksLikeListComma =
        (CommaCountInSentence >= 2 || bHasListConjunctionAhead)
        && !bClauseStarterAhead;

    if (!bLooksLikeListComma)
    {
        return EOffgridAIBoundaryPauseClass::HardBreakPause;
    }
    return EOffgridAIBoundaryPauseClass::SoftListPause;
}

static TCHAR PreferBoundary(TCHAR Existing, TCHAR Candidate)
{
    if (IsHardSentenceBoundary(Candidate))
    {
        return Candidate;
    }
    if (Existing == TCHAR(0))
    {
        return Candidate;
    }
    return Existing;
}

static FString StripCmuVariantSuffix(const FString& In)
{
    int32 ParenIndex = INDEX_NONE;
    if (In.FindChar(TEXT('('), ParenIndex) && ParenIndex > 0)
    {
        return In.Left(ParenIndex);
    }
    return In;
}

static FString StripStressDigits(const FString& Phoneme)
{
    FString Out;
    for (TCHAR C : Phoneme)
    {
        if (!FChar::IsDigit(C))
        {
            Out.AppendChar(C);
        }
    }
    return Out;
}

static bool IsVowelPhonemeBase(const FString& Base)
{
    static const TSet<FString> Vowels = {
        TEXT("AA"), TEXT("AE"), TEXT("AH"), TEXT("AO"), TEXT("AW"), TEXT("AY"),
        TEXT("EH"), TEXT("ER"), TEXT("EY"), TEXT("IH"), TEXT("IY"), TEXT("OW"),
        TEXT("OY"), TEXT("UH"), TEXT("UW")
    };
    return Vowels.Contains(Base);
}

static int32 CountCmuSyllables(const TArray<FString>& Phones)
{
    int32 Count = 0;
    for (const FString& Phone : Phones)
    {
        if (IsVowelPhonemeBase(StripStressDigits(Phone)))
        {
            ++Count;
        }
    }
    return FMath::Max(Count, 1);
}

using FCmuPronunciation = TArray<FString>;
using FCmuPronunciationVariants = TArray<FCmuPronunciation>;

static bool PronunciationsEqual(const FCmuPronunciation& A, const FCmuPronunciation& B)
{
    if (A.Num() != B.Num()) return false;
    for (int32 Index = 0; Index < A.Num(); ++Index)
    {
        if (A[Index] != B[Index]) return false;
    }
    return true;
}

static const TMap<FString, FCmuPronunciationVariants>& GetCmuDictionary()
{
    static const TMap<FString, FCmuPronunciationVariants> Dict = []()
    {
        TMap<FString, FCmuPronunciationVariants> Out;
        const char* RawData = OffgridAILipsyncEmbedded::GetCmudictDictData();
        const int32 RawSize = OffgridAILipsyncEmbedded::GetCmudictDictDataSize();
        if (!RawData || RawSize <= 0)
        {
            return Out;
        }

        FString Text(UTF8_TO_TCHAR(RawData));
        TArray<FString> Lines;
        Text.ParseIntoArrayLines(Lines, false);

        for (FString Line : Lines)
        {
            Line.TrimStartAndEndInline();
            if (Line.IsEmpty() || Line.StartsWith(TEXT(";;;")))
            {
                continue;
            }

            int32 CommentIndex = INDEX_NONE;
            if (Line.FindChar(TEXT('#'), CommentIndex) && CommentIndex >= 0)
            {
                Line = Line.Left(CommentIndex).TrimStartAndEnd();
            }
            if (Line.IsEmpty())
            {
                continue;
            }

            Line.ReplaceInline(TEXT("\t"), TEXT(" "));
            TArray<FString> Parts;
            Line.ParseIntoArray(Parts, TEXT(" "), true);
            if (Parts.Num() < 2)
            {
                continue;
            }

            const FString Word = NormalizeWord(StripCmuVariantSuffix(Parts[0]));
            if (Word.IsEmpty())
            {
                continue;
            }

            TArray<FString> Phones;
            for (int32 I = 1; I < Parts.Num(); ++I)
            {
                FString Phone = Parts[I].TrimStartAndEnd().ToUpper();
                if (!Phone.IsEmpty())
                {
                    Phones.Add(Phone);
                }
            }
            if (Phones.Num() > 0)
            {
                FCmuPronunciationVariants& Variants = Out.FindOrAdd(Word);
                bool bAlreadyPresent = false;
                for (const FCmuPronunciation& Existing : Variants)
                {
                    if (PronunciationsEqual(Existing, Phones))
                    {
                        bAlreadyPresent = true;
                        break;
                    }
                }
                if (!bAlreadyPresent)
                {
                    Variants.Add(MoveTemp(Phones));
                }
            }
        }
        return Out;
    }();
    return Dict;
}

static const TMap<FString, FCmuPronunciation>& GetTtsPronunciationPreferences()
{
    static const TMap<FString, FCmuPronunciation> Preferences = []()
    {
        TMap<FString, FCmuPronunciation> Out;
        for (const FOffgridAITtsPronunciationPreferenceRow& Row : GTtsPronunciationPreferences)
        {
            FCmuPronunciation Phones;
            FString(Row.Phones).ParseIntoArray(Phones, TEXT(" "), true);
            if (Phones.Num() > 0)
            {
                Out.Add(Row.Word, MoveTemp(Phones));
            }
        }
        return Out;
    }();
    return Preferences;
}

static bool SelectCmuPronunciation(
    const FString& Word,
    FCmuPronunciation& OutPhones,
    int32& OutVariantIndex,
    int32& OutVariantCount)
{
    const TMap<FString, FCmuPronunciationVariants>& Dict = GetCmuDictionary();
    const FCmuPronunciationVariants* Variants = Dict.Find(Word);
    if (!Variants || Variants->Num() <= 0)
    {
        return false;
    }

    OutVariantCount = Variants->Num();
    OutVariantIndex = 0;
    if (const FCmuPronunciation* Preferred = GetTtsPronunciationPreferences().Find(Word))
    {
        for (int32 VariantIndex = 0; VariantIndex < Variants->Num(); ++VariantIndex)
        {
            if (PronunciationsEqual((*Variants)[VariantIndex], *Preferred))
            {
                OutVariantIndex = VariantIndex;
                break;
            }
        }
    }
    OutPhones = (*Variants)[OutVariantIndex];
    return true;
}

static bool LookupCmuPronunciation(
    const FString& Word,
    TArray<FString>& OutPhones,
    int32& OutVariantIndex,
    int32& OutVariantCount)
{
    if (SelectCmuPronunciation(Word, OutPhones, OutVariantIndex, OutVariantCount))
    {
        return true;
    }

    // Possessives are often omitted from user text normalization variants.
    if (Word.EndsWith(TEXT("'s")) && Word.Len() > 2)
    {
        const FString Base = Word.LeftChop(2);
        if (SelectCmuPronunciation(Base, OutPhones, OutVariantIndex, OutVariantCount))
        {
            OutPhones.Add(TEXT("Z"));
            return true;
        }
    }
    if (Word.EndsWith(TEXT("s")) && Word.Len() > 1)
    {
        const FString Base = Word.LeftChop(1);
        if (SelectCmuPronunciation(Base, OutPhones, OutVariantIndex, OutVariantCount))
        {
            OutPhones.Add(TEXT("Z"));
            return true;
        }
    }

    // Product dialogue contains invented compounds and names. If both sides
    // are ordinary CMU words, preserve their full phone sequence rather than
    // collapsing the entire unknown token to one generic vowel.
    for (int32 Split = 3; Split <= Word.Len() - 3; ++Split)
    {
        FString Suffix;
        for (int32 Index = Split; Index < Word.Len(); ++Index)
        {
            Suffix.AppendChar(Word[Index]);
        }
        FCmuPronunciation Left;
        FCmuPronunciation Right;
        int32 LeftVariantIndex = 0;
        int32 LeftVariantCount = 0;
        int32 RightVariantIndex = 0;
        int32 RightVariantCount = 0;
        if (SelectCmuPronunciation(Word.Left(Split), Left, LeftVariantIndex, LeftVariantCount)
            && SelectCmuPronunciation(Suffix, Right, RightVariantIndex, RightVariantCount))
        {
            OutPhones = Left;
            for (const FString& Phone : Right)
            {
                OutPhones.Add(Phone);
            }
            OutVariantIndex = 0;
            OutVariantCount = 1;
            return true;
        }
    }

    return false;
}

static FName DefaultPoseFor(EOffgridAITextViseme V)
{
    switch (V)
    {
    case EOffgridAITextViseme::MBP: return TEXT("22_MBP");
    case EOffgridAITextViseme::AAA: return TEXT("07_Aa");
    case EOffgridAITextViseme::EEE: return TEXT("03_Ee");
    case EOffgridAITextViseme::OOO: return TEXT("11_Oo");
    case EOffgridAITextViseme::WUH: return TEXT("12_Ww-Oo-");
    case EOffgridAITextViseme::FVS: return TEXT("20_FV");
    default: return NAME_None;
    }
}

static void AddEvent(TArray<FOffgridAITextVisemeEvent>& Events, EOffgridAITextViseme V, FName PoseID, int32 WordIndex, int32 SpeechRegionIndex, int32 SentenceIndex, const FString& Word, float Strength, FName Generator, float LocalOrder, int32 SourcePhoneIndex = INDEX_NONE, const FString& SourcePhone = FString(), const FString& SourcePhoneBase = FString(), EOffgridAIVisualPhoneRole VisualRole = EOffgridAIVisualPhoneRole::PrimaryPose, bool bIsRenderable = true)
{
    if (V == EOffgridAITextViseme::Rest)
    {
        return;
    }

    const FName ResolvedPose = PoseID.IsNone() ? DefaultPoseFor(V) : PoseID;
    if (ResolvedPose.IsNone())
    {
        return;
    }

    FOffgridAITextVisemeEvent E;
    E.Viseme = V;
    E.PoseID = ResolvedPose;
    E.Strength = Strength;
    E.SourceText = Word;
    E.WordIndex = WordIndex;
    E.SpeechRegionIndex = SpeechRegionIndex;
    E.SentenceIndex = SentenceIndex;
    E.SourcePhoneIndex = SourcePhoneIndex;
    E.SourcePhoneGlobalIndex = INDEX_NONE;
    E.SourcePhone = SourcePhone;
    E.SourcePhoneBase = SourcePhoneBase;
    E.PhoneLocalNorm = LocalOrder;
    E.VisualRole = VisualRole;
    E.bIsRenderable = bIsRenderable;
    E.bIsStrongVisibleEvent = (V == EOffgridAITextViseme::MBP || V == EOffgridAITextViseme::WUH || ResolvedPose == FName(TEXT("20_FV")));
    E.Generator = Generator;
    E.StartNorm = LocalOrder; // Overwritten by the final normalized timing pass.
    E.EndNorm = 0.0f;
    Events.Add(E);
}

static bool IsStress1Or2(const FString& Phone)
{
    return Phone.Contains(TEXT("1")) || Phone.Contains(TEXT("2"));
}

static bool AddPhoneViseme(TArray<FOffgridAITextVisemeEvent>& Events, const FString& Word, const TArray<FString>& WordPhones, const FString& Phone, int32 PhoneIndex, int32 PhoneCount, int32 WordIndex, int32 SpeechRegionIndex, int32 SentenceIndex)
{
    const FString Base = StripStressDigits(Phone);
    const float LocalOrder = (static_cast<float>(PhoneIndex) + 0.5f) / FMath::Max(static_cast<float>(PhoneCount), 1.0f);
    const bool bStressed = IsStress1Or2(Phone);
    const float VowelStrength = bStressed ? 0.92f : 0.50f;

    // Strong visible consonants. These are CMU phonemes, not letters.
    if (Base == TEXT("M") || Base == TEXT("B") || Base == TEXT("P"))
    {
        AddEvent(Events, EOffgridAITextViseme::MBP, TEXT("22_MBP"), WordIndex, SpeechRegionIndex, SentenceIndex, Word, 1.00f, TEXT("cmu_bilabial"), LocalOrder, PhoneIndex, Phone, Base);
        return true;
    }
    if (Base == TEXT("F") || Base == TEXT("V"))
    {
        AddEvent(Events, EOffgridAITextViseme::FVS, TEXT("20_FV"), WordIndex, SpeechRegionIndex, SentenceIndex, Word, 0.96f, TEXT("cmu_fv"), LocalOrder, PhoneIndex, Phone, Base);
        return true;
    }
    if (Base == TEXT("W"))
    {
        AddEvent(Events, EOffgridAITextViseme::WUH, TEXT("12_Ww-Oo-"), WordIndex, SpeechRegionIndex, SentenceIndex, Word, 0.88f, TEXT("cmu_w"), LocalOrder, PhoneIndex, Phone, Base);
        return true;
    }
    if (Base == TEXT("R"))
    {
        AddEvent(Events, EOffgridAITextViseme::OOO, TEXT("17_Rr"), WordIndex, SpeechRegionIndex, SentenceIndex, Word, 0.78f, TEXT("cmu_rhotic"), LocalOrder, PhoneIndex, Phone, Base);
        return true;
    }

    // Consonant articulation poses supported by the MetaHuman viseme library.
    if (Base == TEXT("S") || Base == TEXT("Z"))
    {
        AddEvent(Events, EOffgridAITextViseme::FVS, TEXT("21_FV-Ee-"), WordIndex, SpeechRegionIndex, SentenceIndex, Word, 0.42f, TEXT("cmu_sibilant_teeth_wide"), LocalOrder, PhoneIndex, Phone, Base, EOffgridAIVisualPhoneRole::SupportingPose);
        return true;
    }
    if (Base == TEXT("TH") || Base == TEXT("DH"))
    {
        AddEvent(Events, EOffgridAITextViseme::FVS, TEXT("24_Tongue_Th"), WordIndex, SpeechRegionIndex, SentenceIndex, Word, 0.62f, TEXT("cmu_dental_tongue"), LocalOrder, PhoneIndex, Phone, Base, EOffgridAIVisualPhoneRole::SupportingPose);
        return true;
    }
    if (Base == TEXT("T") || Base == TEXT("D"))
    {
        AddEvent(Events, EOffgridAITextViseme::AAA, TEXT("01_TDS-Ah-"), WordIndex, SpeechRegionIndex, SentenceIndex, Word, 0.40f, TEXT("cmu_alveolar_stop"), LocalOrder, PhoneIndex, Phone, Base, EOffgridAIVisualPhoneRole::SupportingPose);
        return true;
    }
    if (Base == TEXT("L") || Base == TEXT("N"))
    {
        AddEvent(Events, EOffgridAITextViseme::EEE, TEXT("23_Tongue_LNTDS"), WordIndex, SpeechRegionIndex, SentenceIndex, Word, 0.46f, TEXT("cmu_lingual"), LocalOrder, PhoneIndex, Phone, Base, EOffgridAIVisualPhoneRole::SupportingPose);
        return true;
    }
    if (Base == TEXT("K") || Base == TEXT("G"))
    {
        AddEvent(Events, EOffgridAITextViseme::EEE, TEXT("13_KGY_TDS"), WordIndex, SpeechRegionIndex, SentenceIndex, Word, 0.34f, TEXT("cmu_velar_stop"), LocalOrder, PhoneIndex, Phone, Base, EOffgridAIVisualPhoneRole::SupportingPose);
        return true;
    }
    if (Base == TEXT("Y"))
    {
        AddEvent(Events, EOffgridAITextViseme::EEE, TEXT("02_TDS_KGY-Ee-"), WordIndex, SpeechRegionIndex, SentenceIndex, Word, 0.40f, TEXT("cmu_palatal_glide"), LocalOrder, PhoneIndex, Phone, Base, EOffgridAIVisualPhoneRole::SupportingPose);
        return true;
    }
    if (Base == TEXT("CH") || Base == TEXT("JH") || Base == TEXT("SH") || Base == TEXT("ZH"))
    {
        AddEvent(Events, EOffgridAITextViseme::FVS, TEXT("14_ChJjSh"), WordIndex, SpeechRegionIndex, SentenceIndex, Word, 0.42f, TEXT("cmu_affricate_sibilant"), LocalOrder, PhoneIndex, Phone, Base);
        return true;
    }
    // HH borrows the following vowel's visible shape. It remains a complete
    // timing/acoustic phone but does not create an independent jaw-open target.
    if (Base == TEXT("HH"))
    {
        AddEvent(Events, EOffgridAITextViseme::AAA, TEXT("08_Ah"), WordIndex, SpeechRegionIndex, SentenceIndex, Word, 0.0f, TEXT("cmu_hh_timing_waypoint"), LocalOrder, PhoneIndex, Phone, Base, EOffgridAIVisualPhoneRole::Coarticulated, false);
        return false;
    }

    if (Base == TEXT("AY"))
    {
        AddEvent(Events, EOffgridAITextViseme::AAA, TEXT("05_Ay"), WordIndex, SpeechRegionIndex, SentenceIndex, Word, bStressed ? 0.94f : 0.68f, TEXT("cmu_ay"), LocalOrder, PhoneIndex, Phone, Base);
        return true;
    }
    if (Base == TEXT("AA") || Base == TEXT("AW"))
    {
        AddEvent(Events, EOffgridAITextViseme::AAA, TEXT("07_Aa"), WordIndex, SpeechRegionIndex, SentenceIndex, Word, VowelStrength, TEXT("cmu_open_vowel"), LocalOrder, PhoneIndex, Phone, Base);
        return true;
    }
    if (Base == TEXT("AE"))
    {
        AddEvent(Events, EOffgridAITextViseme::AAA, TEXT("08_Ah"), WordIndex, SpeechRegionIndex, SentenceIndex, Word, VowelStrength, TEXT("cmu_ah_vowel"), LocalOrder, PhoneIndex, Phone, Base);
        return true;
    }
    if (Base == TEXT("AH"))
    {
        AddEvent(Events, EOffgridAITextViseme::AAA, TEXT("18_Uh"), WordIndex, SpeechRegionIndex, SentenceIndex, Word, VowelStrength, TEXT("cmu_central_vowel"), LocalOrder, PhoneIndex, Phone, Base);
        return true;
    }
    if (Base == TEXT("ER"))
    {
        AddEvent(Events, EOffgridAITextViseme::OOO, TEXT("10_Or"), WordIndex, SpeechRegionIndex, SentenceIndex, Word, VowelStrength, TEXT("cmu_rhotic_vowel"), LocalOrder, PhoneIndex, Phone, Base);
        return true;
    }
    if (Base == TEXT("EH") || Base == TEXT("EY"))
    {
        AddEvent(Events, EOffgridAITextViseme::AAA, TEXT("06_Eh"), WordIndex, SpeechRegionIndex, SentenceIndex, Word, VowelStrength, TEXT("cmu_eh_vowel"), LocalOrder, PhoneIndex, Phone, Base);
        return true;
    }
    if (Base == TEXT("IH"))
    {
        AddEvent(Events, EOffgridAITextViseme::EEE, TEXT("04_Ih"), WordIndex, SpeechRegionIndex, SentenceIndex, Word, VowelStrength, TEXT("cmu_ih_vowel"), LocalOrder, PhoneIndex, Phone, Base);
        return true;
    }
    if (Base == TEXT("IY"))
    {
        AddEvent(Events, EOffgridAITextViseme::EEE, TEXT("03_Ee"), WordIndex, SpeechRegionIndex, SentenceIndex, Word, VowelStrength, TEXT("cmu_front_vowel"), LocalOrder, PhoneIndex, Phone, Base);
        return true;
    }
    if (Base == TEXT("AO") || Base == TEXT("OW") || Base == TEXT("OY"))
    {
        AddEvent(Events, EOffgridAITextViseme::OOO, TEXT("09_Oh"), WordIndex, SpeechRegionIndex, SentenceIndex, Word, VowelStrength, TEXT("cmu_oh_vowel"), LocalOrder, PhoneIndex, Phone, Base);
        return true;
    }
    if (Base == TEXT("UH"))
    {
        AddEvent(Events, EOffgridAITextViseme::AAA, TEXT("18_Uh"), WordIndex, SpeechRegionIndex, SentenceIndex, Word, VowelStrength, TEXT("cmu_uh_vowel"), LocalOrder, PhoneIndex, Phone, Base);
        return true;
    }
    if (Base == TEXT("UW"))
    {
        AddEvent(Events, EOffgridAITextViseme::OOO, TEXT("11_Oo"), WordIndex, SpeechRegionIndex, SentenceIndex, Word, VowelStrength, TEXT("cmu_oo_vowel"), LocalOrder, PhoneIndex, Phone, Base);
        return true;
    }

    // Remaining phones are timing phones rather than dedicated visible targets.
    return false;
}

static void AddCmuWordVisemeEvents(TArray<FOffgridAITextVisemeEvent>& Events, const FString& Word, const TArray<FString>& Phones, int32 WordIndex, int32 SpeechRegionIndex, int32 SentenceIndex)
{
    for (int32 I = 0; I < Phones.Num(); ++I)
    {
        AddPhoneViseme(Events, Word, Phones, Phones[I], I, Phones.Num(), WordIndex, SpeechRegionIndex, SentenceIndex);
    }
}


static float ExpectedPhoneWeightSeconds(const FString& Base)
{
    // Use corpus-derived consonant duration priors where they helped, but keep
    // vowels deliberately conservative so the text plan does not consume whole
    // regions too early when a speaker stretches nuclei or inserts intra-region
    // pauses.
    if (IsVowelPhonemeBase(Base))
    {
        if (Base == TEXT("AW") || Base == TEXT("AY") || Base == TEXT("EY") || Base == TEXT("OW") || Base == TEXT("OY") || Base == TEXT("UH"))
        {
            return 0.120f;
        }
        if (Base == TEXT("AE") || Base == TEXT("ER") || Base == TEXT("IY"))
        {
            return 0.100f;
        }
        return 0.090f;
    }

    struct FCorpusPhoneDuration
    {
        const TCHAR* Base;
        float Seconds;
    };

    static const FCorpusPhoneDuration CorpusDurations[] = {
        { TEXT("B"), 0.070f },  { TEXT("CH"), 0.130f }, { TEXT("D"), 0.050f },
        { TEXT("DH"), 0.040f }, { TEXT("F"), 0.100f },  { TEXT("G"), 0.060f },
        { TEXT("HH"), 0.070f }, { TEXT("JH"), 0.100f }, { TEXT("K"), 0.080f },
        { TEXT("L"), 0.080f },  { TEXT("M"), 0.070f },  { TEXT("N"), 0.040f },
        { TEXT("NG"), 0.070f }, { TEXT("P"), 0.080f },  { TEXT("R"), 0.070f },
        { TEXT("S"), 0.110f },  { TEXT("SH"), 0.100f }, { TEXT("T"), 0.060f },
        { TEXT("TH"), 0.070f }, { TEXT("V"), 0.060f },  { TEXT("W"), 0.080f },
        { TEXT("Y"), 0.080f },  { TEXT("Z"), 0.070f },  { TEXT("ZH"), 0.090f },
    };

    for (const FCorpusPhoneDuration& Entry : CorpusDurations)
    {
        if (Base == Entry.Base)
        {
            return Entry.Seconds;
        }
    }

    if (Base == TEXT("M") || Base == TEXT("B") || Base == TEXT("P")) return 0.070f;
    if (Base == TEXT("W") || Base == TEXT("R")) return 0.080f;
    if (Base == TEXT("F") || Base == TEXT("V") || Base == TEXT("CH") || Base == TEXT("JH") || Base == TEXT("SH") || Base == TEXT("ZH")) return 0.100f;
    if (Base == TEXT("TH") || Base == TEXT("DH") || Base == TEXT("L") || Base == TEXT("N")) return 0.060f;
    if (Base == TEXT("T") || Base == TEXT("D") || Base == TEXT("K") || Base == TEXT("G") || Base == TEXT("Y") || Base == TEXT("HH")) return 0.060f;
    if (Base == TEXT("S") || Base == TEXT("Z")) return 0.090f;
    return 0.060f;
}

static float ExpectedPauseSeconds(EOffgridAIBoundaryPauseClass PauseClass)
{
    switch (PauseClass)
    {
    case EOffgridAIBoundaryPauseClass::HardBreakPause:
        // Runtime pacing needs a compressed prior rather than the full MFA
        // median, otherwise we over-hold boundaries before occupancy can
        // resolve the actual resume.
        return 0.200f;
    case EOffgridAIBoundaryPauseClass::SoftListPause:
        return 0.040f;
    case EOffgridAIBoundaryPauseClass::None:
    default:
        return 0.0f;
    }
}

static bool PhoneHasVisibleViseme(const TArray<FOffgridAITextVisemeEvent>& Events, int32 WordIndex, int32 SourcePhoneIndex)
{
    for (const FOffgridAITextVisemeEvent& E : Events)
    {
        if (E.WordIndex == WordIndex && E.SourcePhoneIndex == SourcePhoneIndex && E.bIsRenderable) return true;
    }
    return false;
}

static int32 EstimateUnknownWordSyllables(const FString& Word)
{
    int32 Count = 0;
    bool bPrevVowel = false;
    for (int32 I = 0; I < Word.Len(); ++I)
    {
        const TCHAR C = FChar::ToLower(Word[I]);
        const bool bVowel = (C == TEXT('a') || C == TEXT('e') || C == TEXT('i') || C == TEXT('o') || C == TEXT('u') || C == TEXT('y'));
        if (bVowel && !bPrevVowel)
        {
            ++Count;
        }
        bPrevVowel = bVowel;
    }
    return FMath::Max(Count, 1);
}

static void AddConservativeUnknownWordEvents(TArray<FOffgridAITextVisemeEvent>& Events, const FString& Word, int32 WordIndex, int32 SpeechRegionIndex, int32 SentenceIndex)
{
    // Unknown words should not fall back to letter soup. Emit one low-strength
    // generic vowel so timing can continue, and make CMU misses obvious in logs.
    AddEvent(Events, EOffgridAITextViseme::AAA, TEXT("06_Eh"), WordIndex, SpeechRegionIndex, SentenceIndex, Word, 0.28f, TEXT("cmu_miss_conservative_single_vowel"), 0.5f, 0, TEXT("UNK"), TEXT("UNK"));
}
}

FOffgridAITextVisemePlan FOffgridAITextVisemePlanner::BuildPlan(const FText& Dialogue, float CharactersPerSecond, float MinDurationSeconds)
{
    FOffgridAITextVisemePlan Plan;
    const FString Text = Dialogue.ToString();
    TArray<FString> Words;
    TArray<TCHAR> Boundaries;

    FString Current;
    for (int32 TextIndex = 0; TextIndex < Text.Len(); ++TextIndex)
    {
        const TCHAR C = Text[TextIndex];
        if (IsWordTokenCharAt(Text, TextIndex))
        {
            Current.AppendChar(C);
        }
        else
        {
            const bool bSpeechRegionBoundary = IsSpeechRegionBoundaryAt(Text, TextIndex);
            if (!Current.IsEmpty())
            {
                Words.Add(NormalizeWord(Current));
                Boundaries.Add(bSpeechRegionBoundary ? C : TCHAR(0));
                Current.Reset();
            }
            else if (Boundaries.Num() > 0 && bSpeechRegionBoundary)
            {
                Boundaries.Last() = PreferBoundary(Boundaries.Last(), C);
            }
        }
    }
    if (!Current.IsEmpty())
    {
        Words.Add(NormalizeWord(Current));
        Boundaries.Add(TCHAR(0));
    }

    TArray<TArray<FString>> WordPhones;
    TArray<bool> WordCmuHit;
    TArray<int32> WordPronunciationVariantIndices;
    TArray<int32> WordPronunciationVariantCounts;
    TArray<EOffgridAIBoundaryPauseClass> BoundaryPauseClasses;
    Plan.WordPhoneBeginIndices.Init(INDEX_NONE, Words.Num());
    Plan.WordPhoneEndIndices.Init(INDEX_NONE, Words.Num());
    Plan.WordVisibleEventBeginIndices.Init(INDEX_NONE, Words.Num());
    Plan.WordVisibleEventEndIndices.Init(INDEX_NONE, Words.Num());
    int32 Sentence = 0;
    int32 TotalUnits = 0;
    TArray<int32> WordUnits;

    for (int32 W = 0; W < Words.Num(); ++W)
    {
        const FString& Word = Words[W];
        TArray<FString> Phones;
        int32 PronunciationVariantIndex = 0;
        int32 PronunciationVariantCount = 1;
        const bool bCmuHit = LookupCmuPronunciation(
            Word,
            Phones,
            PronunciationVariantIndex,
            PronunciationVariantCount);
        WordPhones.Add(Phones);
        WordCmuHit.Add(bCmuHit);
        WordPronunciationVariantIndices.Add(PronunciationVariantIndex);
        WordPronunciationVariantCounts.Add(PronunciationVariantCount);

        const int32 Syllables = bCmuHit ? CountCmuSyllables(Phones) : EstimateUnknownWordSyllables(Word);
        const int32 VisiblePhones = bCmuHit ? FMath::Max(Phones.Num(), 1) : 1;
        const int32 Units = FMath::Clamp(Syllables + FMath::Clamp(VisiblePhones / 3, 1, 3), 2, 6);
        WordUnits.Add(Units);
        TotalUnits += Units;

        Plan.WordSpeechRegionIndices.Add(0);
        Plan.WordSentenceIndices.Add(Sentence);
        Plan.WordSyllableCounts.Add(Syllables);
        Plan.WordBoundaryPunctuationAfter.Add(Boundaries.IsValidIndex(W) ? Boundaries[W] : TCHAR(0));
        const EOffgridAIBoundaryPauseClass BoundaryPauseClass = ClassifyBoundaryPause(Words, Boundaries, W);
        BoundaryPauseClasses.Add(BoundaryPauseClass);
        Plan.WordBoundaryPauseClassAfter.Add(BoundaryPauseClass);
        Plan.WordBoundaryPauseSecondsAfter.Add(ExpectedPauseSeconds(BoundaryPauseClass));
        const TCHAR B = Boundaries.IsValidIndex(W) ? Boundaries[W] : TCHAR(0);
        if (IsHardSentenceBoundary(B)) { ++Sentence; }
    }

    Sentence = 0;
    int32 UnitCursor = 0;
    for (int32 W = 0; W < Words.Num(); ++W)
    {
        const FString& Word = Words[W];
        if (Word.IsEmpty()) continue;
        const int32 WordStartUnit = UnitCursor;
        const int32 Units = WordUnits.IsValidIndex(W) ? WordUnits[W] : 2;
        const int32 FirstEventIndex = Plan.Events.Num();
        Plan.WordVisibleEventBeginIndices[W] = FirstEventIndex;

        if (WordCmuHit.IsValidIndex(W) && WordCmuHit[W])
        {
            AddCmuWordVisemeEvents(Plan.Events, Word, WordPhones[W], W, 0, Sentence);
        }
        else
        {
            AddConservativeUnknownWordEvents(Plan.Events, Word, W, 0, Sentence);
        }

        const TArray<FString>* PhonesForWordPtr = (WordCmuHit.IsValidIndex(W) && WordCmuHit[W]) ? &WordPhones[W] : nullptr;
        if (PhonesForWordPtr && PhonesForWordPtr->Num() > 0)
        {
            Plan.WordPhoneBeginIndices[W] = Plan.ExpectedPhones.Num();
            for (int32 PIdx = 0; PIdx < PhonesForWordPtr->Num(); ++PIdx)
            {
                const FString Phone = (*PhonesForWordPtr)[PIdx];
                const FString Base = StripStressDigits(Phone);
                FOffgridAIExpectedPhone Expected;
                Expected.PhoneIndex = Plan.ExpectedPhones.Num();
                Expected.WordPhoneIndex = PIdx;
                Expected.Phone = Phone;
                Expected.BasePhone = Base;
                Expected.SourceWord = Word;
                Expected.WordIndex = W;
                Expected.SpeechRegionIndex = 0;
                Expected.SentenceIndex = Sentence;
                Expected.bIsVowel = IsVowelPhonemeBase(Base);
                Expected.PronunciationVariantIndex = WordPronunciationVariantIndices.IsValidIndex(W)
                    ? WordPronunciationVariantIndices[W]
                    : 0;
                Expected.PronunciationVariantCount = WordPronunciationVariantCounts.IsValidIndex(W)
                    ? WordPronunciationVariantCounts[W]
                    : 1;
                Expected.bIsVisibleViseme = PhoneHasVisibleViseme(Plan.Events, W, PIdx);
                if (Expected.bIsVisibleViseme)
                {
                    for (int32 EIdx = FirstEventIndex; EIdx < Plan.Events.Num(); ++EIdx)
                    {
                        const FOffgridAITextVisemeEvent& Event = Plan.Events[EIdx];
                        if (Event.WordIndex == W && Event.SourcePhoneIndex == PIdx)
                        {
                            Expected.VisualRole = Event.VisualRole;
                            break;
                        }
                    }
                }
                else if (Base == TEXT("HH"))
                {
                    Expected.VisualRole = EOffgridAIVisualPhoneRole::Coarticulated;
                }
                Expected.BoundaryAfterWord = Boundaries.IsValidIndex(W) ? Boundaries[W] : TCHAR(0);
                Expected.WeightSeconds = ExpectedPhoneWeightSeconds(Base);
                Plan.ExpectedPhones.Add(Expected);
            }
            Plan.WordPhoneEndIndices[W] = Plan.ExpectedPhones.Num();

            for (int32 EIdx = FirstEventIndex; EIdx < Plan.Events.Num(); ++EIdx)
            {
                FOffgridAITextVisemeEvent& Event = Plan.Events[EIdx];
                if (Event.WordIndex != W || Event.SourcePhoneIndex == INDEX_NONE)
                {
                    continue;
                }

                for (const FOffgridAIExpectedPhone& Expected : Plan.ExpectedPhones)
                {
                    if (Expected.WordIndex == W && Expected.WordPhoneIndex == Event.SourcePhoneIndex)
                    {
                        Event.SourcePhoneGlobalIndex = Expected.PhoneIndex;
                        break;
                    }
                }
            }

            for (int32 PhoneGlobalIndex = Plan.WordPhoneBeginIndices[W]; PhoneGlobalIndex < Plan.WordPhoneEndIndices[W]; ++PhoneGlobalIndex)
            {
                FOffgridAIExpectedPhone& Expected = Plan.ExpectedPhones[PhoneGlobalIndex];
                for (int32 EIdx = FirstEventIndex; EIdx < Plan.Events.Num(); ++EIdx)
                {
                    const FOffgridAITextVisemeEvent& Event = Plan.Events[EIdx];
                    if (Event.WordIndex == W && Event.SourcePhoneGlobalIndex == Expected.PhoneIndex)
                    {
                        Expected.FirstVisibleEventIndex = EIdx;
                        break;
                    }
                }
            }
        }
        else
        {
            Plan.WordPhoneBeginIndices[W] = Plan.ExpectedPhones.Num();
            FOffgridAIExpectedPhone Expected;
            Expected.PhoneIndex = Plan.ExpectedPhones.Num();
            Expected.WordPhoneIndex = 0;
            Expected.Phone = TEXT("UNK");
            Expected.BasePhone = TEXT("UNK");
            Expected.SourceWord = Word;
            Expected.WordIndex = W;
            Expected.SpeechRegionIndex = 0;
            Expected.SentenceIndex = Sentence;
            Expected.bIsVisibleViseme = (Plan.Events.Num() > FirstEventIndex);
            Expected.VisualRole = Expected.bIsVisibleViseme
                ? EOffgridAIVisualPhoneRole::PrimaryPose
                : EOffgridAIVisualPhoneRole::TimingOnly;
            Expected.FirstVisibleEventIndex = Expected.bIsVisibleViseme ? FirstEventIndex : INDEX_NONE;
            Expected.BoundaryAfterWord = Boundaries.IsValidIndex(W) ? Boundaries[W] : TCHAR(0);
            Expected.WeightSeconds = 0.100f;
            Plan.ExpectedPhones.Add(Expected);
            Plan.WordPhoneEndIndices[W] = Plan.ExpectedPhones.Num();

            for (int32 EIdx = FirstEventIndex; EIdx < Plan.Events.Num(); ++EIdx)
            {
                FOffgridAITextVisemeEvent& Event = Plan.Events[EIdx];
                if (Event.WordIndex != W) continue;
                Event.SourcePhoneIndex = 0;
                Event.SourcePhoneGlobalIndex = Plan.WordPhoneBeginIndices[W];
                Event.SourcePhone = TEXT("UNK");
                Event.SourcePhoneBase = TEXT("UNK");
            }
        }

        Plan.WordVisibleEventEndIndices[W] = Plan.Events.Num();

        const int32 EventCountForWord = Plan.Events.Num() - FirstEventIndex;
        for (int32 EIdx = FirstEventIndex; EIdx < Plan.Events.Num(); ++EIdx)
        {
            FOffgridAITextVisemeEvent& E = Plan.Events[EIdx];
            const float Local = FMath::Clamp(E.StartNorm, 0.05f, 0.95f);
            const float CenterUnit = static_cast<float>(WordStartUnit) + Local * static_cast<float>(Units);
            const float CenterNorm = TotalUnits > 0 ? CenterUnit / static_cast<float>(TotalUnits) : 0.0f;
            const float Half = (EventCountForWord > 2 ? 0.32f : 0.42f) / FMath::Max(static_cast<float>(TotalUnits), 1.0f);
            E.StartNorm = FMath::Clamp(CenterNorm - Half, 0.0f, 1.0f);
            E.EndNorm = FMath::Clamp(CenterNorm + Half, 0.0f, 1.0f);
        }

        UnitCursor += Units;
        const TCHAR B = Boundaries.IsValidIndex(W) ? Boundaries[W] : TCHAR(0);
        if (IsHardSentenceBoundary(B)) { ++Sentence; }
    }

    Plan.Events.Sort([](const FOffgridAITextVisemeEvent& A, const FOffgridAITextVisemeEvent& B){ return A.StartNorm < B.StartNorm; });
    Plan.EstimatedDurationSeconds = FMath::Max(MinDurationSeconds, Text.Len() / FMath::Max(CharactersPerSecond, 1.0f));
    return Plan;
}

const TCHAR* FOffgridAITextVisemePlanner::ToPoseKey(EOffgridAITextViseme Viseme)
{
    switch (Viseme)
    {
    case EOffgridAITextViseme::MBP: return TEXT("22_MBP");
    case EOffgridAITextViseme::AAA: return TEXT("07_Aa");
    case EOffgridAITextViseme::EEE: return TEXT("03_Ee");
    case EOffgridAITextViseme::OOO: return TEXT("11_Oo");
    case EOffgridAITextViseme::WUH: return TEXT("12_Ww-Oo-");
    case EOffgridAITextViseme::FVS: return TEXT("20_FV");
    default: return TEXT("Rest");
    }
}

FString FOffgridAITextVisemePlanner::ToDebugString(EOffgridAITextViseme Viseme)
{
    return FString(ToPoseKey(Viseme));
}
