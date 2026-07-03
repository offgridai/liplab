#include "Lipsync/OffgridAITextVisemePlanner.h"
#include "Lipsync/OffgridAICmudictData.h"

namespace
{
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
    return C == TEXT('.') || C == TEXT('!') || C == TEXT('?');
}

static bool IsSoftBoundary(TCHAR C)
{
    return C == TEXT(',') || C == TEXT(';') || C == TEXT(':');
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

static const TMap<FString, TArray<FString>>& GetCmuDictionary()
{
    static const TMap<FString, TArray<FString>> Dict = []()
    {
        TMap<FString, TArray<FString>> Out;
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
            if (Word.IsEmpty() || Out.Contains(Word))
            {
                continue; // Prefer the first CMU pronunciation.
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
                Out.Add(Word, MoveTemp(Phones));
            }
        }
        return Out;
    }();
    return Dict;
}

static bool LookupCmuPronunciation(const FString& Word, TArray<FString>& OutPhones)
{
    const TMap<FString, TArray<FString>>& Dict = GetCmuDictionary();
    if (const TArray<FString>* Found = Dict.Find(Word))
    {
        OutPhones = *Found;
        return true;
    }

    // Possessives are often omitted from user text normalization variants.
    if (Word.EndsWith(TEXT("'s")) && Word.Len() > 2)
    {
        const FString Base = Word.LeftChop(2);
        if (const TArray<FString>* FoundBase = Dict.Find(Base))
        {
            OutPhones = *FoundBase;
            OutPhones.Add(TEXT("Z"));
            return true;
        }
    }
    if (Word.EndsWith(TEXT("s")) && Word.Len() > 1)
    {
        const FString Base = Word.LeftChop(1);
        if (const TArray<FString>* FoundBase = Dict.Find(Base))
        {
            OutPhones = *FoundBase;
            OutPhones.Add(TEXT("Z"));
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

static void AddEvent(TArray<FOffgridAITextVisemeEvent>& Events, EOffgridAITextViseme V, FName PoseID, int32 WordIndex, int32 PhraseIndex, int32 SentenceIsland, const FString& Word, float Strength, FName Generator, float LocalOrder, int32 SourcePhoneIndex = INDEX_NONE, const FString& SourcePhone = FString(), const FString& SourcePhoneBase = FString())
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

    // Collapse exact repeated mouth shapes inside one word. CMU may contain
    // repeated consonants or vowel tails; one held visual target is clearer than
    // multiple identical planned events.
    if (Events.Num() > 0)
    {
        const FOffgridAITextVisemeEvent& Prev = Events.Last();
        if (Prev.WordIndex == WordIndex && Prev.PoseID == ResolvedPose)
        {
            return;
        }
    }

    FOffgridAITextVisemeEvent E;
    E.Viseme = V;
    E.PoseID = ResolvedPose;
    E.Strength = Strength;
    E.SourceText = Word;
    E.WordIndex = WordIndex;
    E.PhraseIndex = PhraseIndex;
    E.SentenceIslandIndex = SentenceIsland;
    E.SourcePhoneIndex = SourcePhoneIndex;
    E.SourcePhoneGlobalIndex = INDEX_NONE;
    E.SourcePhone = SourcePhone;
    E.SourcePhoneBase = SourcePhoneBase;
    E.PhoneLocalNorm = LocalOrder;
    E.bIsStrongVisibleEvent = (V == EOffgridAITextViseme::MBP || V == EOffgridAITextViseme::WUH || ResolvedPose == FName(TEXT("20_FV")) || (ResolvedPose == FName(TEXT("14_ChJjSh")) && Strength >= 0.70f));
    E.Generator = Generator;
    E.StartNorm = LocalOrder; // Overwritten by the final normalized timing pass.
    E.EndNorm = 0.0f;
    Events.Add(E);
}

static bool IsStress1Or2(const FString& Phone)
{
    return Phone.Contains(TEXT("1")) || Phone.Contains(TEXT("2"));
}

static bool WordAlreadyHasPose(const TArray<FOffgridAITextVisemeEvent>& Events, int32 WordIndex, FName PoseID)
{
    for (const FOffgridAITextVisemeEvent& E : Events)
    {
        if (E.WordIndex == WordIndex && E.PoseID == PoseID)
        {
            return true;
        }
    }
    return false;
}

static bool HasStrongVowel(const TArray<FString>& Phones)
{
    for (const FString& Phone : Phones)
    {
        const FString Base = StripStressDigits(Phone);
        if (IsVowelPhonemeBase(Base) && IsStress1Or2(Phone))
        {
            return true;
        }
    }
    return false;
}

static int32 CountVowels(const TArray<FString>& Phones)
{
    int32 Count = 0;
    for (const FString& Phone : Phones)
    {
        if (IsVowelPhonemeBase(StripStressDigits(Phone)))
        {
            ++Count;
        }
    }
    return Count;
}

static bool IsReducedVowelToSuppress(const FString& Base, const FString& Phone, const TArray<FString>& WordPhones)
{
    // This is the important distinction between a CMU phoneme plan and the old
    // letter-soup planner. Reduced IH0/AH0/ER0 tails preserve pronunciation, but
    // they are usually not separate readable mouth targets when a word already
    // has a stressed vowel. Examples: sandwiches, dollars, there.
    if (!HasStrongVowel(WordPhones) || CountVowels(WordPhones) <= 1)
    {
        return false;
    }

    const bool bUnstressed = !(Phone.Contains(TEXT("1")) || Phone.Contains(TEXT("2")));
    if (!bUnstressed)
    {
        return false;
    }

    return Base == TEXT("IH") || Base == TEXT("AH") || Base == TEXT("ER");
}

static bool AddPhoneViseme(TArray<FOffgridAITextVisemeEvent>& Events, const FString& Word, const TArray<FString>& WordPhones, const FString& Phone, int32 PhoneIndex, int32 PhoneCount, int32 WordIndex, int32 Phrase, int32 Sentence)
{
    const FString Base = StripStressDigits(Phone);
    const float LocalOrder = (static_cast<float>(PhoneIndex) + 0.5f) / FMath::Max(static_cast<float>(PhoneCount), 1.0f);
    const bool bStressed = IsStress1Or2(Phone);
    const float VowelStrength = bStressed ? 0.92f : 0.50f;

    // Strong visible consonants. These are CMU phonemes, not letters.
    if (Base == TEXT("M") || Base == TEXT("B") || Base == TEXT("P"))
    {
        if (!WordAlreadyHasPose(Events, WordIndex, TEXT("22_MBP")))
        {
            AddEvent(Events, EOffgridAITextViseme::MBP, TEXT("22_MBP"), WordIndex, Phrase, Sentence, Word, 1.00f, TEXT("cmu_bilabial"), LocalOrder, PhoneIndex, Phone, Base);
        }
        return true;
    }
    if (Base == TEXT("F") || Base == TEXT("V"))
    {
        if (!WordAlreadyHasPose(Events, WordIndex, TEXT("20_FV")))
        {
            AddEvent(Events, EOffgridAITextViseme::FVS, TEXT("20_FV"), WordIndex, Phrase, Sentence, Word, 0.96f, TEXT("cmu_fv"), LocalOrder, PhoneIndex, Phone, Base);
        }
        return true;
    }
    if (Base == TEXT("W"))
    {
        AddEvent(Events, EOffgridAITextViseme::WUH, TEXT("12_Ww-Oo-"), WordIndex, Phrase, Sentence, Word, 0.88f, TEXT("cmu_w"), LocalOrder, PhoneIndex, Phone, Base);
        return true;
    }
    if (Base == TEXT("CH") || Base == TEXT("JH") || Base == TEXT("SH") || Base == TEXT("ZH"))
    {
        if (!WordAlreadyHasPose(Events, WordIndex, TEXT("14_ChJjSh")))
        {
            AddEvent(Events, EOffgridAITextViseme::FVS, TEXT("14_ChJjSh"), WordIndex, Phrase, Sentence, Word, 0.82f, TEXT("cmu_affricate_sibilant"), LocalOrder, PhoneIndex, Phone, Base);
        }
        return true;
    }

    // Do not render S/Z/TH/DH as independent visemes. This was the main source
    // of nonsense plans such as can->Ch, there->FV, sounds->Ch/Aa/Ch, and the
    // extra Ch spam in sandwiches. Keep them in the CMU syllable/timing count,
    // but do not create a visible event.
    if (Base == TEXT("S") || Base == TEXT("Z") || Base == TEXT("TH") || Base == TEXT("DH"))
    {
        return false;
    }

    if (IsReducedVowelToSuppress(Base, Phone, WordPhones))
    {
        return false;
    }

    if (Base == TEXT("AY"))
    {
        AddEvent(Events, EOffgridAITextViseme::AAA, TEXT("05_Ay"), WordIndex, Phrase, Sentence, Word, bStressed ? 0.94f : 0.68f, TEXT("cmu_ay"), LocalOrder, PhoneIndex, Phone, Base);
        return true;
    }
    if (Base == TEXT("AA") || Base == TEXT("AE") || Base == TEXT("AH") || Base == TEXT("AW"))
    {
        AddEvent(Events, EOffgridAITextViseme::AAA, TEXT("07_Aa"), WordIndex, Phrase, Sentence, Word, VowelStrength, TEXT("cmu_open_vowel"), LocalOrder, PhoneIndex, Phone, Base);
        return true;
    }
    if (Base == TEXT("EH") || Base == TEXT("ER") || Base == TEXT("EY"))
    {
        AddEvent(Events, EOffgridAITextViseme::AAA, TEXT("06_Eh"), WordIndex, Phrase, Sentence, Word, VowelStrength, TEXT("cmu_eh_vowel"), LocalOrder, PhoneIndex, Phone, Base);
        return true;
    }
    if (Base == TEXT("IH") || Base == TEXT("IY"))
    {
        AddEvent(Events, EOffgridAITextViseme::EEE, TEXT("03_Ee"), WordIndex, Phrase, Sentence, Word, VowelStrength, TEXT("cmu_front_vowel"), LocalOrder, PhoneIndex, Phone, Base);
        return true;
    }
    if (Base == TEXT("AO") || Base == TEXT("OW") || Base == TEXT("OY"))
    {
        AddEvent(Events, EOffgridAITextViseme::OOO, TEXT("09_Oh"), WordIndex, Phrase, Sentence, Word, VowelStrength, TEXT("cmu_oh_vowel"), LocalOrder, PhoneIndex, Phone, Base);
        return true;
    }
    if (Base == TEXT("UH") || Base == TEXT("UW"))
    {
        AddEvent(Events, EOffgridAITextViseme::OOO, TEXT("11_Oo"), WordIndex, Phrase, Sentence, Word, VowelStrength, TEXT("cmu_oo_vowel"), LocalOrder, PhoneIndex, Phone, Base);
        return true;
    }

    // K/G/T/D/N/L/R/Y/HH/etc. are timing phones, not explicit mouth-shape
    // targets for the MetaHuman viseme set used here.
    return false;
}

static void AddCmuWordVisemeEvents(TArray<FOffgridAITextVisemeEvent>& Events, const FString& Word, const TArray<FString>& Phones, int32 WordIndex, int32 Phrase, int32 Sentence)
{
    const int32 FirstEventIndex = Events.Num();
    for (int32 I = 0; I < Phones.Num(); ++I)
    {
        AddPhoneViseme(Events, Word, Phones, Phones[I], I, Phones.Num(), WordIndex, Phrase, Sentence);
    }

    if (Events.Num() == FirstEventIndex)
    {
        AddEvent(Events, EOffgridAITextViseme::AAA, TEXT("06_Eh"), WordIndex, Phrase, Sentence, Word, 0.22f, TEXT("cmu_no_visible_phone"), 0.5f);
    }
}


static float ExpectedPhoneWeightSeconds(const FString& Base)
{
    if (IsVowelPhonemeBase(Base)) return 0.105f;
    if (Base == TEXT("M") || Base == TEXT("B") || Base == TEXT("P")) return 0.080f;
    if (Base == TEXT("F") || Base == TEXT("V") || Base == TEXT("CH") || Base == TEXT("JH") || Base == TEXT("SH") || Base == TEXT("ZH")) return 0.085f;
    if (Base == TEXT("S") || Base == TEXT("Z") || Base == TEXT("TH") || Base == TEXT("DH")) return 0.060f;
    return 0.055f;
}

static bool PhoneHasVisibleViseme(const TArray<FOffgridAITextVisemeEvent>& Events, int32 WordIndex, int32 SourcePhoneIndex)
{
    for (const FOffgridAITextVisemeEvent& E : Events)
    {
        if (E.WordIndex == WordIndex && E.SourcePhoneIndex == SourcePhoneIndex) return true;
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

static void AddConservativeUnknownWordEvents(TArray<FOffgridAITextVisemeEvent>& Events, const FString& Word, int32 WordIndex, int32 Phrase, int32 Sentence)
{
    // Unknown words should not fall back to letter soup. Emit one low-strength
    // generic vowel so timing can continue, and make CMU misses obvious in logs.
    AddEvent(Events, EOffgridAITextViseme::AAA, TEXT("06_Eh"), WordIndex, Phrase, Sentence, Word, 0.28f, TEXT("cmu_miss_conservative_single_vowel"), 0.5f);
}
}

FOffgridAITextVisemePlan FOffgridAITextVisemePlanner::BuildPlan(const FText& Dialogue, float CharactersPerSecond, float MinDurationSeconds)
{
    FOffgridAITextVisemePlan Plan;
    const FString Text = Dialogue.ToString();
    TArray<FString> Words;
    TArray<TCHAR> Boundaries;

    FString Current;
    for (TCHAR C : Text)
    {
        if (FChar::IsAlnum(C) || C == TCHAR('\''))
        {
            Current.AppendChar(C);
        }
        else
        {
            if (!Current.IsEmpty())
            {
                Words.Add(NormalizeWord(Current));
                Boundaries.Add((IsHardSentenceBoundary(C) || IsSoftBoundary(C)) ? C : TCHAR(0));
                Current.Reset();
            }
            else if (Boundaries.Num() > 0 && (IsHardSentenceBoundary(C) || IsSoftBoundary(C)))
            {
                Boundaries.Last() = C;
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
    int32 Phrase = 0;
    int32 Sentence = 0;
    int32 TotalUnits = 0;
    TArray<int32> WordUnits;

    for (int32 W = 0; W < Words.Num(); ++W)
    {
        const FString& Word = Words[W];
        TArray<FString> Phones;
        const bool bCmuHit = LookupCmuPronunciation(Word, Phones);
        WordPhones.Add(Phones);
        WordCmuHit.Add(bCmuHit);

        const int32 Syllables = bCmuHit ? CountCmuSyllables(Phones) : EstimateUnknownWordSyllables(Word);
        const int32 VisiblePhones = bCmuHit ? FMath::Max(Phones.Num(), 1) : 1;
        const int32 Units = FMath::Clamp(Syllables + FMath::Clamp(VisiblePhones / 3, 1, 3), 2, 6);
        WordUnits.Add(Units);
        TotalUnits += Units;

        Plan.WordSentenceIslandIndices.Add(Sentence);
        Plan.WordPhraseIndices.Add(Phrase);
        Plan.WordSyllableCounts.Add(Syllables);
        Plan.WordBoundaryPunctuationAfter.Add(Boundaries.IsValidIndex(W) ? Boundaries[W] : TCHAR(0));
        const TCHAR B = Boundaries.IsValidIndex(W) ? Boundaries[W] : TCHAR(0);
        if (IsSoftBoundary(B)) ++Phrase;
        if (IsHardSentenceBoundary(B)) { ++Phrase; ++Sentence; }
    }

    Phrase = 0;
    Sentence = 0;
    int32 UnitCursor = 0;
    for (int32 W = 0; W < Words.Num(); ++W)
    {
        const FString& Word = Words[W];
        if (Word.IsEmpty()) continue;
        const int32 WordStartUnit = UnitCursor;
        const int32 Units = WordUnits.IsValidIndex(W) ? WordUnits[W] : 2;
        const int32 FirstEventIndex = Plan.Events.Num();

        if (WordCmuHit.IsValidIndex(W) && WordCmuHit[W])
        {
            AddCmuWordVisemeEvents(Plan.Events, Word, WordPhones[W], W, Phrase, Sentence);
        }
        else
        {
            AddConservativeUnknownWordEvents(Plan.Events, Word, W, Phrase, Sentence);
        }

        const TArray<FString>* PhonesForWordPtr = (WordCmuHit.IsValidIndex(W) && WordCmuHit[W]) ? &WordPhones[W] : nullptr;
        if (PhonesForWordPtr && PhonesForWordPtr->Num() > 0)
        {
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
                Expected.PhraseIndex = Phrase;
                Expected.SentenceIslandIndex = Sentence;
                Expected.bIsVowel = IsVowelPhonemeBase(Base);
                Expected.bIsVisibleViseme = PhoneHasVisibleViseme(Plan.Events, W, PIdx);
                Expected.BoundaryAfterWord = Boundaries.IsValidIndex(W) ? Boundaries[W] : TCHAR(0);
                Expected.WeightSeconds = ExpectedPhoneWeightSeconds(Base);
                Plan.ExpectedPhones.Add(Expected);
            }

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
        }
        else
        {
            FOffgridAIExpectedPhone Expected;
            Expected.PhoneIndex = Plan.ExpectedPhones.Num();
            Expected.Phone = TEXT("UNK");
            Expected.BasePhone = TEXT("UNK");
            Expected.SourceWord = Word;
            Expected.WordIndex = W;
            Expected.PhraseIndex = Phrase;
            Expected.SentenceIslandIndex = Sentence;
            Expected.bIsVisibleViseme = true;
            Expected.BoundaryAfterWord = Boundaries.IsValidIndex(W) ? Boundaries[W] : TCHAR(0);
            Expected.WeightSeconds = 0.100f;
            Plan.ExpectedPhones.Add(Expected);
        }

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
        if (IsSoftBoundary(B)) ++Phrase;
        if (IsHardSentenceBoundary(B)) { ++Phrase; ++Sentence; }
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
