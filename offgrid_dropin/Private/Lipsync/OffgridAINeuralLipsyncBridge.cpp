#include "Lipsync/OffgridAINeuralLipsyncBridge.h"

#include <algorithm>
#include <cmath>

namespace
{
using namespace offgridai::neural_streamer;

static uint32 HashFeatureName(const FString& Value)
{
    uint32 Hash = 2166136261u;
    for (int32 Index = 0; Index < Value.Len(); ++Index)
    {
        Hash ^= static_cast<uint8>(Value[Index]);
        Hash *= 16777619u;
    }
    return Hash;
}

struct FBridgeToken
{
    FString Phone;
    FString Pose;
    int32 EventIndex = INDEX_NONE;
    int32 WordIndex = INDEX_NONE;
    int32 PhoneIndex = INDEX_NONE;
    int32 SentenceIndex = INDEX_NONE;
    float DurationPriorSec = 0.075f;
    float IsVowel = 0.0f;
    float VisualRole = 0.0f;
    float TextCenterNorm = 0.0f;
    float Strength = 0.0f;
    float IsSilence = 0.0f;
    float IsWordStart = 0.0f;
    float IsWordEnd = 0.0f;
    bool IsSentenceBoundary = false;
    bool IsPauseBoundary = false;
    bool IsHardPauseBoundary = false;
    EOffgridAIPunctuationType PunctuationType = EOffgridAIPunctuationType::None;
};

static FBridgeToken SilenceToken(float TextCenterNorm)
{
    FBridgeToken Token;
    Token.Phone = FString(TEXT("SIL"));
    Token.Pose = FString(TEXT("__SILENCE__"));
    Token.DurationPriorSec = 0.040f;
    Token.TextCenterNorm = TextCenterNorm;
    Token.IsSilence = 1.0f;
    return Token;
}
}

FOffgridAINeuralTranscriptTensor FOffgridAINeuralLipsyncBridge::BuildTranscriptTensor(
    const FOffgridAITextVisemePlan& Plan)
{
    using namespace offgridai::neural_streamer;
    TArray<FBridgeToken> Visible;
    for (int32 EventIndex = 0; EventIndex < Plan.Events.Num(); ++EventIndex)
    {
        const auto& Event = Plan.Events[EventIndex];
        if (!Event.bIsRenderable) continue;
        FBridgeToken Token;
        Token.Phone = Event.SourcePhoneBase;
        Token.Pose = Event.PoseID.ToString();
        Token.EventIndex = EventIndex;
        Token.WordIndex = Event.WordIndex;
        Token.PhoneIndex = Event.SourcePhoneGlobalIndex;
        Token.SentenceIndex = Event.SentenceIndex;
        Token.VisualRole = static_cast<float>(Event.VisualRole);
        Token.TextCenterNorm = Event.StartNorm
            + 0.5f * (Event.EndNorm - Event.StartNorm);
        Token.Strength = Event.Strength;
        if (Plan.ExpectedPhones.IsValidIndex(Event.SourcePhoneGlobalIndex))
        {
            const auto& Phone = Plan.ExpectedPhones[Event.SourcePhoneGlobalIndex];
            Token.DurationPriorSec = Phone.WeightSeconds;
            Token.IsVowel = Phone.bIsVowel ? 1.0f : 0.0f;
        }
        Visible.Add(MoveTemp(Token));
    }
    for (int32 Index = 0; Index < Visible.Num(); ++Index)
    {
        Visible[Index].IsWordStart = Index == 0
            || Visible[Index - 1].WordIndex != Visible[Index].WordIndex;
        Visible[Index].IsWordEnd = Index + 1 == Visible.Num()
            || Visible[Index + 1].WordIndex != Visible[Index].WordIndex;
    }

    TArray<FBridgeToken> Sequence;
    if (Visible.Num() > 0)
    {
        Sequence.Add(SilenceToken(0.0f));
        for (int32 Index = 0; Index < Visible.Num(); ++Index)
        {
            Sequence.Add(Visible[Index]);
            if (Index + 1 < Visible.Num()
                && Visible[Index].WordIndex != Visible[Index + 1].WordIndex)
            {
                FBridgeToken Silence = SilenceToken(0.5f
                    * (Visible[Index].TextCenterNorm
                        + Visible[Index + 1].TextCenterNorm));
                Silence.SentenceIndex = Visible[Index + 1].SentenceIndex;
                Silence.IsSentenceBoundary =
                    Visible[Index].SentenceIndex != Visible[Index + 1].SentenceIndex;
                const int32 PreviousWordIndex = Visible[Index].WordIndex;
                Silence.IsPauseBoundary =
                    Plan.WordBoundaryPauseClassAfter.IsValidIndex(PreviousWordIndex)
                    && Plan.WordBoundaryPauseClassAfter[PreviousWordIndex]
                        != EOffgridAIBoundaryPauseClass::None;
                Silence.IsHardPauseBoundary =
                    Plan.WordBoundaryPauseClassAfter.IsValidIndex(PreviousWordIndex)
                    && Plan.WordBoundaryPauseClassAfter[PreviousWordIndex]
                        == EOffgridAIBoundaryPauseClass::HardBreakPause;
                if (Plan.WordBoundaryPunctuationTypesAfter.IsValidIndex(PreviousWordIndex))
                {
                    Silence.PunctuationType =
                        Plan.WordBoundaryPunctuationTypesAfter[PreviousWordIndex];
                }
                if (Silence.IsSentenceBoundary) Silence.DurationPriorSec = 0.120f;
                Sequence.Add(MoveTemp(Silence));
            }
        }
        FBridgeToken Trailing = SilenceToken(1.0f);
        const int32 LastWordIndex = Visible.Last().WordIndex;
        if (Plan.WordBoundaryPunctuationTypesAfter.IsValidIndex(LastWordIndex))
        {
            Trailing.PunctuationType =
                Plan.WordBoundaryPunctuationTypesAfter[LastWordIndex];
        }
        Sequence.Add(MoveTemp(Trailing));
    }

    FOffgridAINeuralTranscriptTensor Result;
    Result.Features.Init(0.0f, Sequence.Num() * kTokenDimensions);
    for (int32 Index = 0; Index < Sequence.Num(); ++Index)
    {
        const FBridgeToken& Token = Sequence[Index];
        float* Row = Result.Features.GetData() + Index * kTokenDimensions;
        Row[HashFeatureName(Token.Phone) % kPhoneBuckets] = 1.0f;
        Row[kPhoneBuckets + HashFeatureName(Token.Pose) % kPoseBuckets] = 1.0f;
        float* Continuous = Row + kPhoneBuckets + kPoseBuckets;
        Continuous[0] = Token.DurationPriorSec;
        Continuous[1] = Token.IsVowel;
        Continuous[2] = Token.VisualRole / 3.0f;
        Continuous[3] = Token.TextCenterNorm;
        Continuous[4] = Token.Strength;
        Continuous[5] = Index == 0 ? 1.0f : 0.0f;
        Continuous[6] = Index + 1 == Sequence.Num() ? 1.0f : 0.0f;
        Continuous[7] = Token.IsSilence;
        Continuous[8] = Token.IsWordStart;
        Continuous[9] = Token.IsWordEnd;
        Continuous[10] = Token.IsPauseBoundary ? 1.0f : 0.0f;
        const int32 PunctuationValue = static_cast<int32>(Token.PunctuationType);
        Continuous[11] = PunctuationValue != 0 ? 1.0f : 0.0f;
        if (PunctuationValue > 0 && PunctuationValue <= kPunctuationTypeCount)
        {
            Continuous[12 + PunctuationValue - 1] = 1.0f;
        }
        Result.EventIndices.Add(Token.EventIndex);
        Result.WordIndices.Add(Token.WordIndex);
        Result.PhoneIndices.Add(Token.PhoneIndex);
        Result.SilenceTokens.Add(Token.IsSilence > 0.5f);
        Result.SentenceBoundaryTokens.Add(Token.IsSentenceBoundary);
        Result.PauseBoundaryTokens.Add(Token.IsPauseBoundary);
        Result.HardPauseBoundaryTokens.Add(Token.IsHardPauseBoundary);
        Result.PunctuationTypeTokens.Add(Token.PunctuationType);
    }
    return Result;
}

void FOffgridAINeuralLipsyncBridge::BuildAudioFeatureVector(
    const FOffgridAIStreamingAudioFeatureFrame& Frame,
    float* OutFeatures)
{
    if (!OutFeatures) return;
    const float Values[offgridai::neural_streamer::kAudioFeatureCount] = {
        std::log(std::max(Frame.RMS, 1.0e-6f)),
        Frame.RMSNorm,
        Frame.DeltaRMS,
        Frame.Flux,
        Frame.ZCR,
        Frame.LowBandNorm,
        Frame.MidBandNorm,
        Frame.HighBandNorm,
        Frame.SpectralCentroidNorm,
        Frame.Periodicity,
        Frame.RichLowBandNorm,
        Frame.RichMidBandNorm,
        Frame.RichHighBandNorm,
        Frame.RichSpectralCentroidNorm,
        Frame.RichSpectralRolloffNorm,
        Frame.RichSpectralFlatness,
        Frame.RichSpectralFlux,
        Frame.RichPeriodicity,
    };
    for (int32 Index = 0; Index < offgridai::neural_streamer::kAudioFeatureCount; ++Index)
        OutFeatures[Index] = Values[Index];
}
