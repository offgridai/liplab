#include "Lipsync/OffgridAITextVisemePlanner.h"
#include "Lipsync/OffgridAICmudictData.h"

namespace
{
#include "OffgridAITtsPronunciationPreferences.inl"
#include "OffgridAIPhoneDurationPriors.inl"

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

static bool IsAllDigits(const FString& Text)
{
    if (Text.IsEmpty()) return false;
    for (TCHAR C : Text)
    {
        if (!FChar::IsDigit(C)) return false;
    }
    return true;
}

static void AddUnderOneHundredWords(int32 Value, TArray<FString>& OutWords)
{
    static const TCHAR* Ones[] = {
        TEXT("zero"), TEXT("one"), TEXT("two"), TEXT("three"), TEXT("four"),
        TEXT("five"), TEXT("six"), TEXT("seven"), TEXT("eight"), TEXT("nine"),
        TEXT("ten"), TEXT("eleven"), TEXT("twelve"), TEXT("thirteen"), TEXT("fourteen"),
        TEXT("fifteen"), TEXT("sixteen"), TEXT("seventeen"), TEXT("eighteen"), TEXT("nineteen")
    };
    static const TCHAR* Tens[] = {
        TEXT(""), TEXT(""), TEXT("twenty"), TEXT("thirty"), TEXT("forty"),
        TEXT("fifty"), TEXT("sixty"), TEXT("seventy"), TEXT("eighty"), TEXT("ninety")
    };
    if (Value < 20)
    {
        OutWords.Add(FString(Ones[Value]));
        return;
    }
    OutWords.Add(FString(Tens[Value / 10]));
    if ((Value % 10) != 0)
    {
        OutWords.Add(FString(Ones[Value % 10]));
    }
}

static void AddUnderOneThousandWords(int32 Value, TArray<FString>& OutWords)
{
    if (Value >= 100)
    {
        AddUnderOneHundredWords(Value / 100, OutWords);
        OutWords.Add(TEXT("hundred"));
        Value %= 100;
    }
    if (Value > 0)
    {
        AddUnderOneHundredWords(Value, OutWords);
    }
}

static bool ParsePositiveInt(const FString& Digits, int32& OutValue)
{
    if (!IsAllDigits(Digits) || Digits.Len() > 9) return false;
    int64 Value = 0;
    for (TCHAR C : Digits)
    {
        Value = Value * 10 + static_cast<int64>(C - TEXT('0'));
        if (Value > 999999) return false;
    }
    OutValue = static_cast<int32>(Value);
    return true;
}

static bool ExpandCardinalNumber(const FString& Digits, TArray<FString>& OutWords)
{
    int32 Value = 0;
    if (!ParsePositiveInt(Digits, Value)) return false;
    if (Value == 0)
    {
        OutWords.Add(TEXT("zero"));
        return true;
    }
    if (Value >= 1000)
    {
        AddUnderOneThousandWords(Value / 1000, OutWords);
        OutWords.Add(TEXT("thousand"));
        Value %= 1000;
    }
    if (Value > 0)
    {
        AddUnderOneThousandWords(Value, OutWords);
    }
    return true;
}

static bool ExpandDecade(const FString& Token, TArray<FString>& OutWords)
{
    if (Token.Len() != 5 || Token[4] != TEXT('s')) return false;
    const FString Digits = Token.Left(4);
    int32 Year = 0;
    if (!ParsePositiveInt(Digits, Year) || Year < 1000 || (Year % 10) != 0) return false;

    AddUnderOneHundredWords(Year / 100, OutWords);
    static const TCHAR* Decades[] = {
        TEXT(""), TEXT("tens"), TEXT("twenties"), TEXT("thirties"), TEXT("forties"),
        TEXT("fifties"), TEXT("sixties"), TEXT("seventies"), TEXT("eighties"), TEXT("nineties")
    };
    OutWords.Add(FString(Decades[(Year % 100) / 10]));
    return true;
}

static void AddNormalizedToken(
    const FString& RawToken,
    TCHAR Boundary,
    TArray<FString>& OutWords,
    TArray<TCHAR>& OutBoundaries)
{
    const FString Token = NormalizeWord(RawToken);
    TArray<FString> Expanded;
    const bool bExpanded = ExpandDecade(Token, Expanded)
        || ExpandCardinalNumber(Token, Expanded);
    if (!bExpanded)
    {
        Expanded.Add(Token);
    }
    for (int32 Index = 0; Index < Expanded.Num(); ++Index)
    {
        OutWords.Add(Expanded[Index]);
        OutBoundaries.Add(Index + 1 == Expanded.Num() ? Boundary : TCHAR(0));
    }
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

static EOffgridAIPunctuationType ClassifyPunctuationType(TCHAR C)
{
    switch (C)
    {
    case TCHAR(0): return EOffgridAIPunctuationType::None;
    case TEXT(','): return EOffgridAIPunctuationType::Comma;
    case TEXT('.'): return EOffgridAIPunctuationType::Period;
    case TEXT('?'): return EOffgridAIPunctuationType::QuestionMark;
    case TEXT('!'): return EOffgridAIPunctuationType::ExclamationMark;
    case TEXT(':'): return EOffgridAIPunctuationType::Colon;
    case TEXT(';'): return EOffgridAIPunctuationType::Semicolon;
    case TEXT('-'): return EOffgridAIPunctuationType::Dash;
    default:
        // The remaining recognized hard boundaries are Unicode en/em dashes.
        return IsHardSentenceBoundary(C)
            ? EOffgridAIPunctuationType::Dash
            : EOffgridAIPunctuationType::Other;
    }
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
    if (FChar::IsAlnum(C) || C == TCHAR('\'')) return true;
    // Grouping separators inside a number are lexical, not punctuation fences.
    return (C == TEXT(',') || C == TEXT('.'))
        && Index > 0
        && Index + 1 < Text.Len()
        && FChar::IsDigit(Text[Index - 1])
        && FChar::IsDigit(Text[Index + 1]);
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

static void BuildPlannedSyllables(FOffgridAITextVisemePlan& Plan)
{
    Plan.Syllables.Reset();
    for (int32 WordIndex = 0; WordIndex < Plan.WordPhoneBeginIndices.Num(); ++WordIndex)
    {
        const int32 WordBegin = Plan.WordPhoneBeginIndices[WordIndex];
        const int32 WordEnd = Plan.WordPhoneEndIndices.IsValidIndex(WordIndex)
            ? Plan.WordPhoneEndIndices[WordIndex]
            : WordBegin;
        TArray<int32> Nuclei;
        for (int32 PhoneIndex = WordBegin; PhoneIndex < WordEnd; ++PhoneIndex)
        {
            if (Plan.ExpectedPhones.IsValidIndex(PhoneIndex) && Plan.ExpectedPhones[PhoneIndex].bIsVowel)
                Nuclei.Add(PhoneIndex);
        }

        if (Plan.WordSyllableCounts.IsValidIndex(WordIndex))
            Plan.WordSyllableCounts[WordIndex] = Nuclei.Num();
        for (int32 LocalIndex = 0; LocalIndex < Nuclei.Num(); ++LocalIndex)
        {
            const int32 Begin = LocalIndex == 0
                ? WordBegin
                : (Nuclei[LocalIndex - 1] + Nuclei[LocalIndex] + 1) / 2;
            const int32 End = LocalIndex + 1 == Nuclei.Num()
                ? WordEnd
                : (Nuclei[LocalIndex] + Nuclei[LocalIndex + 1] + 1) / 2;
            const FOffgridAIExpectedPhone& Nucleus = Plan.ExpectedPhones[Nuclei[LocalIndex]];
            FOffgridAIPlannedSyllable Syllable;
            Syllable.SyllableIndex = Plan.Syllables.Num();
            Syllable.WordSyllableIndex = LocalIndex;
            Syllable.WordIndex = WordIndex;
            Syllable.SpeechRegionIndex = Nucleus.SpeechRegionIndex;
            Syllable.SentenceIndex = Nucleus.SentenceIndex;
            Syllable.PhoneBeginIndex = Begin;
            Syllable.PhoneEndIndex = End;
            Syllable.NucleusPhoneIndex = Nuclei[LocalIndex];
            Plan.Syllables.Add(Syllable);
        }
    }
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

static bool IsVisuallyCoherentPronunciation(const FCmuPronunciation& Phones)
{
    // A standalone R immediately followed by the already-rhotic ER vowel
    // double-articulates the same rhotic gesture (for example F R ER0). MFA
    // may select such an alternate while fitting acoustics, but it is not a
    // coherent transcript-owned visual sequence.
    for (int32 Index = 0; Index + 1 < Phones.Num(); ++Index)
    {
        if (StripStressDigits(Phones[Index]) == TEXT("R")
            && StripStressDigits(Phones[Index + 1]) == TEXT("ER"))
        {
            return false;
        }
    }
    return true;
}

static bool SelectCmuPronunciation(
    const FString& Word,
    const FString& PreviousWord,
    const FString& NextWord,
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
            if (PronunciationsEqual((*Variants)[VariantIndex], *Preferred)
                && IsVisuallyCoherentPronunciation(*Preferred))
            {
                OutVariantIndex = VariantIndex;
                break;
            }
        }
    }

    const FOffgridAITtsContextPronunciationPreferenceRow* BestContext = nullptr;
    for (const FOffgridAITtsContextPronunciationPreferenceRow& Row : GTtsContextPronunciationPreferences)
    {
        if (Word != Row.Word) continue;
        const FString& Neighbor = Row.bPreviousNeighbor ? PreviousWord : NextWord;
        if (Neighbor != Row.NeighborWord) continue;
        FCmuPronunciation ContextPhones;
        FString(Row.Phones).ParseIntoArray(ContextPhones, TEXT(" "), true);
        if (!IsVisuallyCoherentPronunciation(ContextPhones)) continue;
        if (BestContext == nullptr
            || Row.PreferredShare > BestContext->PreferredShare
            || (Row.PreferredShare == BestContext->PreferredShare
                && Row.ObservationCount > BestContext->ObservationCount))
        {
            BestContext = &Row;
        }
    }
    if (BestContext != nullptr)
    {
        FCmuPronunciation ContextPhones;
        FString(BestContext->Phones).ParseIntoArray(ContextPhones, TEXT(" "), true);
        for (int32 VariantIndex = 0; VariantIndex < Variants->Num(); ++VariantIndex)
        {
            if (PronunciationsEqual((*Variants)[VariantIndex], ContextPhones))
            {
                OutVariantIndex = VariantIndex;
                break;
            }
        }
    }
    OutPhones = (*Variants)[OutVariantIndex];
    return true;
}

static bool AppendPhones(const TCHAR* Phones, FCmuPronunciation& OutPhones)
{
    TArray<FString> Parsed;
    FString(Phones).ParseIntoArray(Parsed, TEXT(" "), true);
    for (const FString& Phone : Parsed) OutPhones.Add(Phone);
    return Parsed.Num() > 0;
}

static const TCHAR* LetterNamePhones(TCHAR Letter)
{
    switch (Letter)
    {
    case TEXT('a'): return TEXT("EY1"); case TEXT('b'): return TEXT("B IY1");
    case TEXT('c'): return TEXT("S IY1"); case TEXT('d'): return TEXT("D IY1");
    case TEXT('e'): return TEXT("IY1"); case TEXT('f'): return TEXT("EH1 F");
    case TEXT('g'): return TEXT("JH IY1"); case TEXT('h'): return TEXT("EY1 CH");
    case TEXT('i'): return TEXT("AY1"); case TEXT('j'): return TEXT("JH EY1");
    case TEXT('k'): return TEXT("K EY1"); case TEXT('l'): return TEXT("EH1 L");
    case TEXT('m'): return TEXT("EH1 M"); case TEXT('n'): return TEXT("EH1 N");
    case TEXT('o'): return TEXT("OW1"); case TEXT('p'): return TEXT("P IY1");
    case TEXT('q'): return TEXT("K Y UW1"); case TEXT('r'): return TEXT("AA1 R");
    case TEXT('s'): return TEXT("EH1 S"); case TEXT('t'): return TEXT("T IY1");
    case TEXT('u'): return TEXT("Y UW1"); case TEXT('v'): return TEXT("V IY1");
    case TEXT('w'): return TEXT("D AH1 B AH0 L Y UW0"); case TEXT('x'): return TEXT("EH1 K S");
    case TEXT('y'): return TEXT("W AY1"); case TEXT('z'): return TEXT("Z IY1");
    default: return nullptr;
    }
}

static const TCHAR* SingleGraphemePhones(TCHAR Letter, TCHAR NextLetter)
{
    if (Letter == TEXT('c') && (NextLetter == TEXT('e') || NextLetter == TEXT('i') || NextLetter == TEXT('y'))) return TEXT("S");
    if (Letter == TEXT('g') && (NextLetter == TEXT('e') || NextLetter == TEXT('i') || NextLetter == TEXT('y'))) return TEXT("JH");
    switch (Letter)
    {
    case TEXT('a'): return TEXT("AH0"); case TEXT('b'): return TEXT("B");
    case TEXT('c'): return TEXT("K"); case TEXT('d'): return TEXT("D");
    case TEXT('e'): return TEXT("EH1"); case TEXT('f'): return TEXT("F");
    case TEXT('g'): return TEXT("G"); case TEXT('h'): return TEXT("HH");
    case TEXT('i'): return TEXT("IH1"); case TEXT('j'): return TEXT("JH");
    case TEXT('k'): return TEXT("K"); case TEXT('l'): return TEXT("L");
    case TEXT('m'): return TEXT("M"); case TEXT('n'): return TEXT("N");
    case TEXT('o'): return TEXT("OW1"); case TEXT('p'): return TEXT("P");
    case TEXT('q'): return TEXT("K"); case TEXT('r'): return TEXT("R");
    case TEXT('s'): return TEXT("S"); case TEXT('t'): return TEXT("T");
    case TEXT('u'): return TEXT("AH0"); case TEXT('v'): return TEXT("V");
    case TEXT('w'): return TEXT("W"); case TEXT('x'): return TEXT("K S");
    case TEXT('y'): return TEXT("IY0"); case TEXT('z'): return TEXT("Z");
    default: return nullptr;
    }
}

static bool BuildFallbackPronunciation(const FString& Word, FCmuPronunciation& OutPhones)
{
    OutPhones.Reset();
    bool bAllDigits = !Word.IsEmpty();
    for (TCHAR C : Word) bAllDigits = bAllDigits && FChar::IsDigit(C);
    if (bAllDigits)
    {
        static const TCHAR* DigitWords[] = {
            TEXT("zero"), TEXT("one"), TEXT("two"), TEXT("three"), TEXT("four"),
            TEXT("five"), TEXT("six"), TEXT("seven"), TEXT("eight"), TEXT("nine")};
        for (TCHAR C : Word)
        {
            FCmuPronunciation DigitPhones;
            int32 VariantIndex = 0;
            int32 VariantCount = 0;
            if (!SelectCmuPronunciation(
                DigitWords[C - TEXT('0')], FString(), FString(), DigitPhones, VariantIndex, VariantCount)) return false;
            for (const FString& Phone : DigitPhones) OutPhones.Add(Phone);
        }
        return OutPhones.Num() > 0;
    }

    FString Letters;
    for (TCHAR C : Word)
    {
        if ((C >= TEXT('a') && C <= TEXT('z')) || (C >= TEXT('A') && C <= TEXT('Z')))
        {
            Letters.AppendChar(C >= TEXT('A') && C <= TEXT('Z') ? C - TEXT('A') + TEXT('a') : C);
        }
    }
    if (Letters.Len() < 2) return false;

    bool bHasVowelLetter = false;
    for (TCHAR C : Letters)
    {
        bHasVowelLetter = bHasVowelLetter || C == TEXT('a') || C == TEXT('e')
            || C == TEXT('i') || C == TEXT('o') || C == TEXT('u');
    }
    if (!bHasVowelLetter && Letters.Len() <= 5)
    {
        for (TCHAR C : Letters)
        {
            const TCHAR* Phones = LetterNamePhones(C);
            if (!Phones) return false;
            AppendPhones(Phones, OutPhones);
        }
        return OutPhones.Num() > 0;
    }

    struct FGraphemeRule { const TCHAR* Grapheme; const TCHAR* Phones; };
    static const FGraphemeRule Rules[] = {
        {TEXT("ough"), TEXT("AO1")}, {TEXT("eigh"), TEXT("EY1")},
        {TEXT("tion"), TEXT("SH AH0 N")}, {TEXT("sion"), TEXT("ZH AH0 N")},
        {TEXT("ch"), TEXT("CH")}, {TEXT("sh"), TEXT("SH")}, {TEXT("th"), TEXT("TH")},
        {TEXT("ph"), TEXT("F")}, {TEXT("wh"), TEXT("W")}, {TEXT("ng"), TEXT("NG")},
        {TEXT("qu"), TEXT("K W")}, {TEXT("ck"), TEXT("K")}, {TEXT("ee"), TEXT("IY1")},
        {TEXT("ea"), TEXT("IY1")}, {TEXT("ai"), TEXT("EY1")}, {TEXT("ay"), TEXT("EY1")},
        {TEXT("oa"), TEXT("OW1")}, {TEXT("ow"), TEXT("AW1")}, {TEXT("oo"), TEXT("UW1")},
        {TEXT("ou"), TEXT("AW1")}, {TEXT("oi"), TEXT("OY1")}, {TEXT("oy"), TEXT("OY1")},
        {TEXT("er"), TEXT("ER0")}, {TEXT("ar"), TEXT("AA1 R")}, {TEXT("or"), TEXT("AO1 R")},
        {TEXT("ir"), TEXT("ER1")}, {TEXT("ur"), TEXT("ER1")}};

    for (int32 Index = 0; Index < Letters.Len();)
    {
        bool bMatched = false;
        for (const FGraphemeRule& Rule : Rules)
        {
            const FString Grapheme(Rule.Grapheme);
            bool bStartsHere = Index + Grapheme.Len() <= Letters.Len();
            for (int32 Offset = 0; bStartsHere && Offset < Grapheme.Len(); ++Offset)
            {
                bStartsHere = Letters[Index + Offset] == Grapheme[Offset];
            }
            if (bStartsHere)
            {
                AppendPhones(Rule.Phones, OutPhones);
                Index += Grapheme.Len();
                bMatched = true;
                break;
            }
        }
        if (bMatched) continue;
        const TCHAR Next = Index + 1 < Letters.Len() ? Letters[Index + 1] : TCHAR(0);
        if (const TCHAR* Phones = SingleGraphemePhones(Letters[Index], Next)) AppendPhones(Phones, OutPhones);
        ++Index;
    }

    bool bHasVowelPhone = false;
    for (const FString& Phone : OutPhones) bHasVowelPhone = bHasVowelPhone || IsVowelPhonemeBase(StripStressDigits(Phone));
    if (!bHasVowelPhone) OutPhones.Add(TEXT("AH0"));
    return OutPhones.Num() > 0;
}

static bool LookupCmuPronunciation(
    const FString& Word,
    const FString& PreviousWord,
    const FString& NextWord,
    TArray<FString>& OutPhones,
    int32& OutVariantIndex,
    int32& OutVariantCount)
{
    if (SelectCmuPronunciation(Word, PreviousWord, NextWord, OutPhones, OutVariantIndex, OutVariantCount))
    {
        return true;
    }

    // Possessives are often omitted from user text normalization variants.
    if (Word.EndsWith(TEXT("'s")) && Word.Len() > 2)
    {
        const FString Base = Word.LeftChop(2);
        if (SelectCmuPronunciation(Base, FString(), FString(), OutPhones, OutVariantIndex, OutVariantCount))
        {
            OutPhones.Add(TEXT("Z"));
            return true;
        }
    }
    if (Word.EndsWith(TEXT("s")) && Word.Len() > 1)
    {
        const FString Base = Word.LeftChop(1);
        if (SelectCmuPronunciation(Base, FString(), FString(), OutPhones, OutVariantIndex, OutVariantCount))
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
        if (SelectCmuPronunciation(
                Word.Left(Split), FString(), FString(), Left, LeftVariantIndex, LeftVariantCount)
            && SelectCmuPronunciation(
                Suffix, FString(), FString(), Right, RightVariantIndex, RightVariantCount))
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
    E.bIsStrongVisibleEvent = (
        V == EOffgridAITextViseme::MBP
        || V == EOffgridAITextViseme::WUH
        || V == EOffgridAITextViseme::OOO
        || ResolvedPose == FName(TEXT("20_FV"))
        || ResolvedPose == FName(TEXT("24_Tongue_Th")));
    E.Generator = Generator;
    E.StartNorm = LocalOrder; // Overwritten by the final normalized timing pass.
    E.EndNorm = 0.0f;
    Events.Add(E);
}

static bool AddPhoneViseme(TArray<FOffgridAITextVisemeEvent>& Events, const FString& Word, const TArray<FString>& WordPhones, const FString& Phone, int32 PhoneIndex, int32 PhoneCount, int32 WordIndex, int32 SpeechRegionIndex, int32 SentenceIndex)
{
    const FString Base = StripStressDigits(Phone);
    const float LocalOrder = (static_cast<float>(PhoneIndex) + 0.5f) / FMath::Max(static_cast<float>(PhoneCount), 1.0f);
    // Every vowel is a syllable nucleus and therefore a dominant visible pose.
    // Strength is the final presentation magnitude; the performer does not
    // apply a second pose-specific emphasis policy.
    constexpr float VowelStrength = 1.00f;

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
        AddEvent(Events, EOffgridAITextViseme::FVS, TEXT("24_Tongue_Th"), WordIndex, SpeechRegionIndex, SentenceIndex, Word, 0.82f, TEXT("cmu_dental_tongue_landmark"), LocalOrder, PhoneIndex, Phone, Base);
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
    // /h/ has no independent oral constriction, but it is not visually silent:
    // the jaw is partially open and the lips are already approaching the next
    // vowel. Keep it as a low-strength, transcript-owned neural state so words
    // such as "hello" and "how" have an alignable visible attack instead of
    // beginning late on their first vowel.
    if (Base == TEXT("HH"))
    {
        AddEvent(Events, EOffgridAITextViseme::AAA, TEXT("08_Ah"), WordIndex, SpeechRegionIndex, SentenceIndex, Word, 0.52f, TEXT("cmu_hh_open_attack"), LocalOrder, PhoneIndex, Phone, Base, EOffgridAIVisualPhoneRole::Coarticulated);
        return true;
    }

    if (Base == TEXT("AY"))
    {
        AddEvent(Events, EOffgridAITextViseme::AAA, TEXT("05_Ay"), WordIndex, SpeechRegionIndex, SentenceIndex, Word, VowelStrength, TEXT("cmu_ay"), LocalOrder, PhoneIndex, Phone, Base);
        return true;
    }
    if (Base == TEXT("AW"))
    {
        // Keep one authoritative acoustic token. The performer expresses the
        // phone's open-to-round motion inside this event so decoder topology
        // and downstream word timing remain unchanged.
        AddEvent(Events, EOffgridAITextViseme::AAA, TEXT("07_Aa"), WordIndex, SpeechRegionIndex, SentenceIndex, Word, VowelStrength, TEXT("cmu_aw_diphthong"), LocalOrder, PhoneIndex, Phone, Base);
        return true;
    }
    if (Base == TEXT("AA"))
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
        if (Word == TEXT("to"))
        {
            AddEvent(Events, EOffgridAITextViseme::OOO, TEXT("11_Oo"), WordIndex, SpeechRegionIndex, SentenceIndex, Word, VowelStrength, TEXT("lexical_to_rounding"), LocalOrder, PhoneIndex, Phone, Base);
            return true;
        }
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
        if (Word == TEXT("to"))
        {
            // Conversational TTS frequently reduces "to" to IH/AH. Retain the
            // acoustic phone for alignment, but preserve the word's defining
            // lip-rounding target so the function word remains readable.
            AddEvent(Events, EOffgridAITextViseme::OOO, TEXT("11_Oo"), WordIndex, SpeechRegionIndex, SentenceIndex, Word, VowelStrength, TEXT("lexical_to_rounding"), LocalOrder, PhoneIndex, Phone, Base);
            return true;
        }
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


static float ExpectedPhoneWeightSeconds(const FString& Base, int32 WordPhoneIndex, int32 WordPhoneCount)
{
    const int32 WordPosition = WordPhoneCount <= 1
        ? 0
        : (WordPhoneIndex <= 0 ? 1 : (WordPhoneIndex + 1 >= WordPhoneCount ? 3 : 2));
    float BasePrior = 0.075f;
    for (const FOffgridAIPhoneDurationPriorRow& Row : GPhoneDurationPriors)
    {
        if (Base != Row.BasePhone) continue;
        if (Row.WordPosition == WordPosition)
        {
            return Row.Seconds;
        }
        if (Row.WordPosition == -1)
        {
            BasePrior = Row.Seconds;
        }
    }
    return BasePrior;
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
    // Unknown words should not fall back to letter soup. Emit one conservative
    // but readable syllable nucleus so timing can continue and the mouth does
    // not disappear merely because CMU lacks the word.
    AddEvent(Events, EOffgridAITextViseme::AAA, TEXT("06_Eh"), WordIndex, SpeechRegionIndex, SentenceIndex, Word, 0.78f, TEXT("cmu_miss_conservative_single_vowel"), 0.5f, 0, TEXT("UNK"), TEXT("UNK"));
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
                AddNormalizedToken(
                    Current,
                    bSpeechRegionBoundary ? C : TCHAR(0),
                    Words,
                    Boundaries);
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
        AddNormalizedToken(Current, TCHAR(0), Words, Boundaries);
    }

    TArray<TArray<FString>> WordPhones;
    TArray<bool> WordHasPronunciation;
    TArray<bool> WordUsesFallbackPronunciation;
    TArray<int32> WordPronunciationVariantIndices;
    TArray<int32> WordPronunciationVariantCounts;
    TArray<EOffgridAIBoundaryPauseClass> BoundaryPauseClasses;
    Plan.WordPhoneBeginIndices.Init(INDEX_NONE, Words.Num());
    Plan.WordPhoneEndIndices.Init(INDEX_NONE, Words.Num());
    Plan.WordVisibleEventBeginIndices.Init(INDEX_NONE, Words.Num());
    Plan.WordVisibleEventEndIndices.Init(INDEX_NONE, Words.Num());
    int32 Sentence = 0;
    int32 SpeechRegion = 0;
    int32 TotalUnits = 0;
    TArray<int32> WordUnits;

    for (int32 W = 0; W < Words.Num(); ++W)
    {
        const FString& Word = Words[W];
        const FString PreviousWord = W > 0 ? Words[W - 1] : TEXT("<s>");
        const FString NextWord = W + 1 < Words.Num() ? Words[W + 1] : TEXT("</s>");
        TArray<FString> Phones;
        int32 PronunciationVariantIndex = 0;
        int32 PronunciationVariantCount = 1;
        const bool bCmuHit = LookupCmuPronunciation(
            Word,
            PreviousWord,
            NextWord,
            Phones,
            PronunciationVariantIndex,
            PronunciationVariantCount);
        const bool bUsesFallback = !bCmuHit && BuildFallbackPronunciation(Word, Phones);
        const bool bHasPronunciation = bCmuHit || bUsesFallback;
        WordPhones.Add(Phones);
        WordHasPronunciation.Add(bHasPronunciation);
        WordUsesFallbackPronunciation.Add(bUsesFallback);
        WordPronunciationVariantIndices.Add(PronunciationVariantIndex);
        WordPronunciationVariantCounts.Add(PronunciationVariantCount);

        const int32 Syllables = bHasPronunciation ? CountCmuSyllables(Phones) : EstimateUnknownWordSyllables(Word);
        const int32 VisiblePhones = bHasPronunciation ? FMath::Max(Phones.Num(), 1) : 1;
        const int32 Units = FMath::Clamp(Syllables + FMath::Clamp(VisiblePhones / 3, 1, 3), 2, 6);
        WordUnits.Add(Units);
        TotalUnits += Units;

        Plan.WordSpeechRegionIndices.Add(SpeechRegion);
        Plan.WordSentenceIndices.Add(Sentence);
        Plan.WordSyllableCounts.Add(Syllables);
        Plan.WordBoundaryPunctuationAfter.Add(Boundaries.IsValidIndex(W) ? Boundaries[W] : TCHAR(0));
        Plan.WordBoundaryPunctuationTypesAfter.Add(ClassifyPunctuationType(
            Boundaries.IsValidIndex(W) ? Boundaries[W] : TCHAR(0)));
        const EOffgridAIBoundaryPauseClass BoundaryPauseClass = ClassifyBoundaryPause(Words, Boundaries, W);
        BoundaryPauseClasses.Add(BoundaryPauseClass);
        Plan.WordBoundaryPauseClassAfter.Add(BoundaryPauseClass);
        Plan.WordBoundaryPauseSecondsAfter.Add(ExpectedPauseSeconds(BoundaryPauseClass));
        const TCHAR B = Boundaries.IsValidIndex(W) ? Boundaries[W] : TCHAR(0);
        if (BoundaryPauseClass == EOffgridAIBoundaryPauseClass::HardBreakPause
            && W + 1 < Words.Num())
        {
            ++SpeechRegion;
        }
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
        const int32 WordSpeechRegion = Plan.WordSpeechRegionIndices.IsValidIndex(W)
            ? Plan.WordSpeechRegionIndices[W]
            : 0;
        Plan.WordVisibleEventBeginIndices[W] = FirstEventIndex;

        if (WordHasPronunciation.IsValidIndex(W) && WordHasPronunciation[W])
        {
            AddCmuWordVisemeEvents(
                Plan.Events, Word, WordPhones[W], W, WordSpeechRegion, Sentence);
            if (WordUsesFallbackPronunciation.IsValidIndex(W) && WordUsesFallbackPronunciation[W])
            {
                for (int32 EIdx = FirstEventIndex; EIdx < Plan.Events.Num(); ++EIdx)
                {
                    Plan.Events[EIdx].Generator = TEXT("fallback_g2p");
                }
            }
        }
        else
        {
            AddConservativeUnknownWordEvents(
                Plan.Events, Word, W, WordSpeechRegion, Sentence);
        }

        const TArray<FString>* PhonesForWordPtr = (WordHasPronunciation.IsValidIndex(W) && WordHasPronunciation[W]) ? &WordPhones[W] : nullptr;
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
                Expected.SpeechRegionIndex = WordSpeechRegion;
                Expected.SentenceIndex = Sentence;
                Expected.bIsVowel = IsVowelPhonemeBase(Base);
                Expected.PronunciationVariantIndex = WordPronunciationVariantIndices.IsValidIndex(W)
                    ? WordPronunciationVariantIndices[W]
                    : 0;
                Expected.PronunciationVariantCount = WordPronunciationVariantCounts.IsValidIndex(W)
                    ? WordPronunciationVariantCounts[W]
                    : 1;
                Expected.bUsesFallbackPronunciation = WordUsesFallbackPronunciation.IsValidIndex(W)
                    && WordUsesFallbackPronunciation[W];
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
                Expected.WeightSeconds = ExpectedPhoneWeightSeconds(Base, PIdx, PhonesForWordPtr->Num());
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
            Expected.SpeechRegionIndex = WordSpeechRegion;
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
    BuildPlannedSyllables(Plan);
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
