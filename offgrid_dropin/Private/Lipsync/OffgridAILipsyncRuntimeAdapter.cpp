#include "Lipsync/OffgridAILipsyncRuntimeAdapter.h"

#include "Lipsync/OffgridAITextVisemePlanner.h"
#include "Lipsync/OffgridAIOnlinePhoneAligner.h"

namespace
{
static float EventCenterNorm(const FOffgridAITextVisemeEvent& E)
{
    return FMath::Clamp((E.StartNorm + E.EndNorm) * 0.5f, 0.0f, 1.0f);
}

static FName ResolvePose(const FOffgridAITextVisemeEvent& E)
{
    return E.PoseID.IsNone() ? FName(FOffgridAITextVisemePlanner::ToPoseKey(E.Viseme)) : E.PoseID;
}

static bool IsPose(FName PoseID, const TCHAR* Literal)
{
    return PoseID == FName(Literal);
}

static bool IsVowelPose(FName PoseID)
{
    return IsPose(PoseID, TEXT("07_Aa")) || IsPose(PoseID, TEXT("06_Eh")) || IsPose(PoseID, TEXT("03_Ee"))
        || IsPose(PoseID, TEXT("09_Oh")) || IsPose(PoseID, TEXT("11_Oo")) || IsPose(PoseID, TEXT("12_Ww-Oo-"));
}

static bool IsStrongPose(FName PoseID)
{
    return IsPose(PoseID, TEXT("22_MBP")) || IsPose(PoseID, TEXT("20_FV")) || IsPose(PoseID, TEXT("14_ChJjSh"))
        || IsPose(PoseID, TEXT("12_Ww-Oo-"));
}

static float SpanForPose(FName PoseID)
{
    if (IsPose(PoseID, TEXT("22_MBP"))) return 0.105f;
    if (IsPose(PoseID, TEXT("20_FV"))) return 0.100f;
    if (IsPose(PoseID, TEXT("14_ChJjSh"))) return 0.095f;
    if (IsVowelPose(PoseID)) return 0.135f;
    return 0.110f;
}

static float LeadForPose(FName PoseID)
{
    if (IsPose(PoseID, TEXT("22_MBP"))) return 0.035f;
    if (IsPose(PoseID, TEXT("20_FV"))) return 0.025f;
    if (IsPose(PoseID, TEXT("14_ChJjSh"))) return 0.018f;
    return 0.006f;
}

static float StrengthForEvent(const FOffgridAITextVisemeEvent& E, FName PoseID)
{
    float S = FMath::Clamp(E.Strength, 0.18f, 1.0f);
    if (IsStrongPose(PoseID)) S = FMath::Max(S, 0.78f);
    else if (IsVowelPose(PoseID)) S = FMath::Max(S, 0.58f);
    return FMath::Clamp(S, 0.16f, 1.0f);
}

static int32 FirstUncommittedEventIndex(const FOffgridAIAlignedVisemeTrack& Track)
{
    int32 Next = 0;
    for (const FOffgridAIAlignedVisemeEvent& E : Track.Events)
    {
        Next = FMath::Max(Next, E.EventIndex + 1);
    }
    return Next;
}

static float LastCommittedCenter(const FOffgridAIAlignedVisemeTrack& Track)
{
    return Track.Events.Num() > 0 ? Track.Events.Last().FinalRenderCenterSeconds : 0.0f;
}

static float FirstConfirmedSpeechStart(const TArray<FOffgridAIStreamingSpeechIsland>* Islands)
{
    if (!Islands || Islands->Num() == 0)
    {
        return -1.0f;
    }

    float First = TNumericLimits<float>::Max();
    for (const FOffgridAIStreamingSpeechIsland& Island : *Islands)
    {
        if (Island.AudioBufferEndSec > Island.AudioBufferStartSec)
        {
            First = FMath::Min(First, Island.AudioBufferStartSec);
        }
    }
    return First == TNumericLimits<float>::Max() ? -1.0f : First;
}

static bool SameSentenceAsLast(const FOffgridAIAlignedVisemeTrack& Track, const FOffgridAITextVisemeEvent& T)
{
    if (Track.Events.Num() == 0)
    {
        return false;
    }
    const FOffgridAIAlignedVisemeEvent& Last = Track.Events.Last();
    return Last.SentenceIndex != INDEX_NONE && T.SentenceIslandIndex != INDEX_NONE && Last.SentenceIndex == T.SentenceIslandIndex;
}

static float MaxWallClockGapForPose(FName PoseID)
{
    // Research intent: this is an HMM-style state-exit prior, not a post-hoc
    // island scheduler. During one text sentence, a visible phone state may not
    // occupy an arbitrarily long wall-clock interval just because the acoustic
    // evidence is weak or sparse.
    if (IsPose(PoseID, TEXT("22_MBP")) || IsPose(PoseID, TEXT("20_FV")) || IsPose(PoseID, TEXT("14_ChJjSh")))
    {
        return 0.220f;
    }
    if (IsVowelPose(PoseID))
    {
        return 0.300f;
    }
    return 0.260f;
}


static TCHAR BoundaryPunctuationBeforeEvent(const FOffgridAITextVisemePlan& Plan, int32 EventIndex)
{
    if (!Plan.Events.IsValidIndex(EventIndex))
    {
        return TCHAR(0);
    }

    const int32 WordIndex = Plan.Events[EventIndex].WordIndex;
    const int32 BoundaryWordIndex = WordIndex - 1;
    return Plan.WordBoundaryPunctuationAfter.IsValidIndex(BoundaryWordIndex)
        ? Plan.WordBoundaryPunctuationAfter[BoundaryWordIndex]
        : TCHAR(0);
}

static bool IsSoftCommaBoundary(TCHAR Punctuation)
{
    return Punctuation == TCHAR(',');
}

static bool FindNextSpeechIslandStartAfter(const TArray<FOffgridAIStreamingSpeechIsland>* Islands, float AfterSec, float MinGapSec, float& OutStartSec)
{
    if (!Islands)
    {
        return false;
    }

    float Best = TNumericLimits<float>::Max();
    for (const FOffgridAIStreamingSpeechIsland& Island : *Islands)
    {
        const float Start = Island.AudioBufferStartSec;
        const float End = FMath::Max(Island.AudioBufferLastSpeechSec, Island.AudioBufferEndSec);
        if (End <= Start)
        {
            continue;
        }
        if (Start > AfterSec + MinGapSec && Start < Best)
        {
            Best = Start;
        }
    }

    if (Best == TNumericLimits<float>::Max())
    {
        return false;
    }
    OutStartSec = Best;
    return true;
}


static float RuntimeSpeechEvidenceScore(const FOffgridAIStreamingAudioFeatureFrame& F)
{
    return FMath::Clamp(FMath::Max(F.RMSNorm, FMath::Max(F.Flux * 0.55f, F.Periodicity * 0.45f)), 0.0f, 1.0f);
}

static bool RuntimeSpeechFrame(const FOffgridAIStreamingAudioFeatureFrame& F)
{
    return RuntimeSpeechEvidenceScore(F) >= 0.055f;
}

static bool RuntimeLowEvidenceFrame(const FOffgridAIStreamingAudioFeatureFrame& F)
{
    // This is deliberately stricter than the detector island close threshold.
    // Detector islands can bridge low-energy valleys; phrase anchoring needs to
    // see those valleys so comma/list boundaries do not leak into the prior word.
    return RuntimeSpeechEvidenceScore(F) < 0.040f && F.RMSNorm < 0.070f;
}

static bool FindNextFeatureSpeechOnsetAfter(
    const TArray<FOffgridAIStreamingAudioFeatureFrame>* Frames,
    float AfterSec,
    float MinSilentGapSec,
    float MaxSearchSec,
    float& OutGapStartSec,
    float& OutSpeechStartSec)
{
    if (!Frames || Frames->Num() == 0)
    {
        return false;
    }

    const float SearchEnd = AfterSec + FMath::Max(MaxSearchSec, MinSilentGapSec + 0.050f);
    bool bInLowRun = false;
    float LowRunStart = 0.0f;
    float LowAccum = 0.0f;
    bool bGapAccepted = false;
    float CandidateSpeechStart = -1.0f;
    float SpeechAccum = 0.0f;

    for (const FOffgridAIStreamingAudioFeatureFrame& F : *Frames)
    {
        if (F.AudioBufferEndSec <= AfterSec)
        {
            continue;
        }
        if (F.AudioBufferStartSec > SearchEnd)
        {
            break;
        }

        const float FrameDuration = FMath::Max(F.AudioBufferEndSec - F.AudioBufferStartSec, 0.0f);
        if (FrameDuration <= 0.0f)
        {
            continue;
        }

        if (!bGapAccepted)
        {
            if (RuntimeLowEvidenceFrame(F))
            {
                if (!bInLowRun)
                {
                    bInLowRun = true;
                    LowRunStart = F.AudioBufferStartSec;
                    LowAccum = 0.0f;
                }
                LowAccum += FrameDuration;
                if (LowAccum >= MinSilentGapSec)
                {
                    bGapAccepted = true;
                    OutGapStartSec = LowRunStart;
                }
            }
            else
            {
                bInLowRun = false;
                LowAccum = 0.0f;
            }
            continue;
        }

        if (RuntimeSpeechFrame(F))
        {
            if (CandidateSpeechStart < 0.0f)
            {
                CandidateSpeechStart = F.AudioBufferStartSec;
                SpeechAccum = 0.0f;
            }
            SpeechAccum += FrameDuration;
            if (SpeechAccum >= 0.025f)
            {
                OutSpeechStartSec = CandidateSpeechStart;
                return true;
            }
        }
        else if (!RuntimeLowEvidenceFrame(F))
        {
            // Weak ambiguous frames neither prove speech nor reset a confirmed gap.
            CandidateSpeechStart = -1.0f;
            SpeechAccum = 0.0f;
        }
    }

    return false;
}



static bool HasSustainedRuntimeSpeechBetween(
    const TArray<FOffgridAIStreamingAudioFeatureFrame>* Frames,
    float StartSec,
    float EndSec,
    float MinSpeechSec)
{
    if (!Frames || EndSec <= StartSec + 0.020f)
    {
        return false;
    }

    float SpeechAccum = 0.0f;
    float LongestRun = 0.0f;
    float CurrentRun = 0.0f;
    for (const FOffgridAIStreamingAudioFeatureFrame& F : *Frames)
    {
        if (F.AudioBufferEndSec <= StartSec)
        {
            continue;
        }
        if (F.AudioBufferStartSec >= EndSec)
        {
            break;
        }

        const float OverlapStart = FMath::Max(F.AudioBufferStartSec, StartSec);
        const float OverlapEnd = FMath::Min(F.AudioBufferEndSec, EndSec);
        const float Dt = FMath::Max(OverlapEnd - OverlapStart, 0.0f);
        if (Dt <= 0.0f)
        {
            continue;
        }

        if (RuntimeSpeechFrame(F))
        {
            SpeechAccum += Dt;
            CurrentRun += Dt;
            LongestRun = FMath::Max(LongestRun, CurrentRun);
        }
        else if (RuntimeLowEvidenceFrame(F))
        {
            CurrentRun = 0.0f;
        }
        // Ambiguous frames do not reset the run; they are common inside vowels
        // and should not make sustained speech look like a pause.
    }

    return SpeechAccum >= MinSpeechSec || LongestRun >= FMath::Min(MinSpeechSec, 0.120f);
}

static bool HasLongRuntimeSilenceBetween(
    const TArray<FOffgridAIStreamingAudioFeatureFrame>* Frames,
    float StartSec,
    float EndSec,
    float MinSilenceSec)
{
    if (!Frames || EndSec <= StartSec + 0.020f)
    {
        return false;
    }

    float LowRun = 0.0f;
    for (const FOffgridAIStreamingAudioFeatureFrame& F : *Frames)
    {
        if (F.AudioBufferEndSec <= StartSec)
        {
            continue;
        }
        if (F.AudioBufferStartSec >= EndSec)
        {
            break;
        }

        const float OverlapStart = FMath::Max(F.AudioBufferStartSec, StartSec);
        const float OverlapEnd = FMath::Min(F.AudioBufferEndSec, EndSec);
        const float Dt = FMath::Max(OverlapEnd - OverlapStart, 0.0f);
        if (Dt <= 0.0f)
        {
            continue;
        }

        if (RuntimeLowEvidenceFrame(F))
        {
            LowRun += Dt;
            if (LowRun >= MinSilenceSec)
            {
                return true;
            }
        }
        else if (RuntimeSpeechFrame(F))
        {
            LowRun = 0.0f;
        }
    }
    return false;
}

static bool HasVisibleAlignedPhone(const FOffgridAIOnlinePhoneAlignmentResult& A, int32 PhoneIndex)
{
    return A.PhoneCenterSeconds.IsValidIndex(PhoneIndex) && A.PhoneCenterSeconds[PhoneIndex] >= 0.0f;
}
}

void FOffgridAILipsyncRuntimeSession::Reset()
{
    NPCID = NAME_None;
    LineID = NAME_None;
    DialogueText.Reset();
    PrerollSec = 0.350f;
    PlaybackSec = 0.0f;
    bBegun = false;
    bPlaybackStarted = false;
    bCommittedTrackBuilt = false;
    bInputStreamClosed = false;
    TextPlan = FOffgridAITextVisemePlan();
    Detector.Reset();
    CommittedTrack = FOffgridAIAlignedVisemeTrack();
    AudioOccupancyDiagnosticRows.Reset();
    AudioOccupancyDiagnosticUpdateOrdinal = 0;
    StreamTailDiagnosticRow = FOffgridAIStreamTailDiagnosticRow();
    PCMChunkCount = 0;
    PCMBytesReceived = 0;
    PCMSamplesReceived = 0;
    LastPCMChunkSampleRate = 0;
    LastPCMChunkChannels = 0;
    LastPCMChunkStartSample = -1;
    LastPCMChunkEndSample = -1;
}

void FOffgridAILipsyncRuntimeSession::BeginLine(const FOffgridAILipsyncRuntimeBeginInput& Input)
{
    Reset();
    NPCID = Input.NPCID;
    LineID = Input.LineID;
    DialogueText = Input.DialogueText;
    PrerollSec = FMath::Max(Input.PrerollSec, 0.0f);
    TextPlan = FOffgridAITextVisemePlanner::BuildPlan(FText::FromString(DialogueText));
    CommittedTrack.NPCID = NPCID;
    CommittedTrack.LineID = LineID;
    bBegun = true;
}

void FOffgridAILipsyncRuntimeSession::PushAudioPCM16(const TArray<uint8>& PCMChunk, int32 BytesToUse, int32 SampleRate, int32 NumChannels, int64 ChunkStartSample)
{
    if (!bBegun) return;
    Detector.AppendPCM16(PCMChunk, BytesToUse, SampleRate, NumChannels, ChunkStartSample);
    ++PCMChunkCount;
    PCMBytesReceived += FMath::Max(BytesToUse, 0);
    LastPCMChunkSampleRate = SampleRate;
    LastPCMChunkChannels = NumChannels;
    const int32 Frames = NumChannels > 0 ? BytesToUse / FMath::Max(NumChannels * static_cast<int32>(sizeof(int16)), 1) : 0;
    PCMSamplesReceived += Frames;
    LastPCMChunkStartSample = ChunkStartSample;
    LastPCMChunkEndSample = ChunkStartSample >= 0 ? ChunkStartSample + Frames : -1;
}

void FOffgridAILipsyncRuntimeSession::CloseInputStream()
{
    bInputStreamClosed = true;
    Detector.Finalize();
}

void FOffgridAILipsyncRuntimeSession::Update(float CurrentPlaybackSec)
{
    PlaybackSec = FMath::Max(CurrentPlaybackSec, 0.0f);
    UpdatePlaybackGate(Detector.GetObservedAudioBufferEndSec());

    FOffgridAILipsyncRuntimeUpdateInput Input;
    Input.TextPlan = &TextPlan;
    Input.SpeechIslands = &Detector.GetIslands();
    Input.AudioFeatureFrames = &Detector.GetFeatureFrames();
    Input.CurrentPlaybackSec = PlaybackSec;
    Input.PrerollSec = PrerollSec;
    Input.ObservedAudioBufferEndSec = Detector.GetObservedAudioBufferEndSec();
    Input.bInputStreamClosed = bInputStreamClosed;
    Input.NPCID = NPCID;
    Input.LineID = LineID;
    FOffgridAILipsyncRuntimeAdapter::UpdateCommittedTrack(Input, CommittedTrack, bCommittedTrackBuilt);
    RecordAlignmentDiagnostics(PlaybackSec, false);
}

void FOffgridAILipsyncRuntimeSession::Finalize(float FinalPlaybackSec)
{
    PlaybackSec = FMath::Max(FinalPlaybackSec, PlaybackSec);
    bInputStreamClosed = true;
    Detector.Finalize(Detector.GetObservedAudioBufferEndSec());
    FOffgridAILipsyncRuntimeUpdateInput Input;
    Input.TextPlan = &TextPlan;
    Input.SpeechIslands = &Detector.GetIslands();
    Input.AudioFeatureFrames = &Detector.GetFeatureFrames();
    Input.CurrentPlaybackSec = PlaybackSec;
    Input.PrerollSec = PrerollSec;
    Input.ObservedAudioBufferEndSec = Detector.GetObservedAudioBufferEndSec();
    Input.bInputStreamClosed = true;
    Input.bPlaybackFinalized = true; // Final drain: represent every remaining planned viseme inside observed speech when possible.
    Input.NPCID = NPCID;
    Input.LineID = LineID;
    FOffgridAILipsyncRuntimeAdapter::UpdateCommittedTrack(Input, CommittedTrack, bCommittedTrackBuilt);
    RecordAlignmentDiagnostics(PlaybackSec, true);
    bCommittedTrackBuilt = true;
}

void FOffgridAILipsyncRuntimeSession::UpdatePlaybackGate(float ObservedEndSec)
{
    if (!bPlaybackStarted && ObservedEndSec >= FMath::Max(PrerollSec, 0.0f))
    {
        bPlaybackStarted = true;
    }
}

void FOffgridAILipsyncRuntimeSession::RecordAlignmentDiagnostics(float CurrentPlaybackSec, bool bFinalReplay)
{
    StreamTailDiagnosticRow.LineID = LineID;
    StreamTailDiagnosticRow.PCMChunkCount = PCMChunkCount;
    StreamTailDiagnosticRow.PCMBytesReceived = PCMBytesReceived;
    StreamTailDiagnosticRow.PCMSamplesReceived = PCMSamplesReceived;
    StreamTailDiagnosticRow.LastSampleRate = LastPCMChunkSampleRate;
    StreamTailDiagnosticRow.LastNumChannels = LastPCMChunkChannels;
    StreamTailDiagnosticRow.LastChunkStartSample = LastPCMChunkStartSample;
    StreamTailDiagnosticRow.LastChunkEndSample = LastPCMChunkEndSample;
    StreamTailDiagnosticRow.ObservedAudioBufferEndSec = Detector.GetObservedAudioBufferEndSec();
    StreamTailDiagnosticRow.FirstSpeechAudioBufferStartSec = Detector.GetFirstSpeechAudioBufferStartSec();
    StreamTailDiagnosticRow.SpeechIslandCount = Detector.GetIslands().Num();
    StreamTailDiagnosticRow.bInputStreamClosed = bInputStreamClosed;
    StreamTailDiagnosticRow.DiagnosticKind = FName(TEXT("runtime_stream_tail"));

    AudioOccupancyDiagnosticRows.Reset();
    for (const FOffgridAIAlignedVisemeEvent& E : CommittedTrack.Events)
    {
        FOffgridAIAudioOccupancyDiagnosticRow R;
        R.LineID = LineID;
        R.UpdateOrdinal = AudioOccupancyDiagnosticUpdateOrdinal;
        R.bFinalReplay = bFinalReplay;
        R.CurrentPlaybackSec = CurrentPlaybackSec;
        R.PrerollSec = PrerollSec;
        R.SourceEventIndex = E.EventIndex;
        R.Word = E.SourceWord;
        R.PoseID = E.PoseID;
        R.PlannedCenterSec = E.TextDiagnosticCenterSeconds;
        R.CommittedCenterSec = E.FinalRenderCenterSeconds;
        R.RenderStartSec = E.RenderStartSeconds;
        R.RenderEndSec = E.RenderEndSeconds;
        R.CommitReason = E.CommitReason;
        R.PlaybackMode = E.CommitReason;
        R.AlignedPhoneStartSec = E.AlignedPhoneStartSeconds;
        R.AlignedPhoneEndSec = E.AlignedPhoneEndSeconds;
        R.AudioActiveSec = E.ObservedActiveElapsedSeconds;
        R.TextPlayheadSec = static_cast<float>(E.EventIndex + 1);
        R.RequiredActiveElapsedSec = E.RequiredActiveElapsedSeconds;
        R.ObservedActiveElapsedSec = E.ObservedActiveElapsedSeconds;
        R.ActiveProgressDeficitSec = E.ActiveProgressDeficitSeconds;
        R.RequiredProgressNorm = E.RequiredProgressNorm;
        R.ObservedProgressNorm = E.ObservedProgressNorm;
        R.ActiveProgressRatio = E.ActiveProgressRatio;
        R.bMappedToObservedSpeech = E.bMappedToObservedSpeech;
        R.DiagnosticKind = E.CommitReason;
        AudioOccupancyDiagnosticRows.Add(R);
    }
    ++AudioOccupancyDiagnosticUpdateOrdinal;
}

void FOffgridAILipsyncRuntimeAdapter::UpdateCommittedTrack(const FOffgridAILipsyncRuntimeUpdateInput& Input, FOffgridAIAlignedVisemeTrack& InOutTrack, bool& bInOutTrackBuilt)
{
    if (!Input.TextPlan)
    {
        return;
    }

    const FOffgridAITextVisemePlan& Plan = *Input.TextPlan;
    InOutTrack.NPCID = Input.NPCID;
    InOutTrack.LineID = Input.LineID;
    if (Plan.Events.Num() == 0)
    {
        bInOutTrackBuilt = true;
        return;
    }

    FOffgridAIOnlinePhoneAlignmentInput AlignmentInput;
    AlignmentInput.Plan = &Plan;
    AlignmentInput.AudioFeatureFrames = Input.AudioFeatureFrames;
    AlignmentInput.SpeechIslands = Input.SpeechIslands;
    AlignmentInput.ObservedAudioEndSec = Input.ObservedAudioBufferEndSec;
    AlignmentInput.PlaybackSec = Input.CurrentPlaybackSec;
    AlignmentInput.LookaheadSec = FMath::Max(Input.PrerollSec, 0.0f);
    AlignmentInput.CommitLagSec = 0.120f;
    // Input-stream close means all audio evidence for this line is known.
    // Final drain is allowed to use transcript-duration fallback so weak or
    // under-resolved acoustic evidence never suppresses planned visemes.
    AlignmentInput.bFinal = Input.bInputStreamClosed || Input.bPlaybackFinalized;
    const FOffgridAIOnlinePhoneAlignmentResult Alignment = FOffgridAIOnlinePhoneAligner::Compute(AlignmentInput);

    const int32 FirstNewIndex = FirstUncommittedEventIndex(InOutTrack);
    float LastCenter = LastCommittedCenter(InOutTrack);

    // Bounded commit lead. Late sentence events remain provisional until playback
    // is near them. Once the TTS stream closes, all future audio is technically
    // observed, but committing the full tail immediately makes conversational
    // endings rigid and prevents later ticks from using fresher playback-local
    // context. Keep final drain streaming-like; at actual playback finalization
    // allow a full drain.
    constexpr float MaxStreamingCommitLeadSec = 0.800f;
    const float PlaybackLeadHorizon = Input.CurrentPlaybackSec + MaxStreamingCommitLeadSec;
    const float PrerollHorizon = Input.CurrentPlaybackSec + FMath::Max(Input.PrerollSec, 0.0f);
    const float RawObservedHorizon = (Input.bInputStreamClosed || Input.bPlaybackFinalized)
        ? FMath::Max(PrerollHorizon, Input.ObservedAudioBufferEndSec + 0.250f)
        : PrerollHorizon;
    const float CommitTimeHorizon = Input.bPlaybackFinalized
        ? RawObservedHorizon
        : FMath::Min(RawObservedHorizon, PlaybackLeadHorizon);
    const float FirstSpeechStart = FirstConfirmedSpeechStart(Input.SpeechIslands);

    for (int32 EventIndex = FirstNewIndex; EventIndex < Plan.Events.Num(); ++EventIndex)
    {
        const FOffgridAITextVisemeEvent& T = Plan.Events[EventIndex];
        const int32 PhoneIndex = FOffgridAIOnlinePhoneAligner::FindPhoneForEvent(Plan, T);
        const bool bHasPhoneEvidence = PhoneIndex != INDEX_NONE && HasVisibleAlignedPhone(Alignment, PhoneIndex);

        const FName Pose = ResolvePose(T);
        float Center = EventCenterNorm(T) * FMath::Max(Plan.EstimatedDurationSeconds, 0.001f);
        FName PlacementReason = FName(TEXT("duration_fallback"));
        if (bHasPhoneEvidence)
        {
            Center = Alignment.PhoneCenterSeconds[PhoneIndex] - LeadForPose(Pose);
            PlacementReason = FName(TEXT("streaming_forced_alignment"));
        }

        // The first visible state must not pre-open before confirmed speech.
        // Render windows can begin before their center, so protect the first
        // event by requiring its center to land far enough inside the first
        // confirmed speech span.
        const float EventSpan = SpanForPose(Pose);
        float BoundaryGuardMinCenter = -1.0f;
        FName BoundaryGuardReason = NAME_None;
        float TailGuardMaxCenter = -1.0f;
        FName TailGuardReason = NAME_None;

        if (EventIndex == 0 && FirstSpeechStart >= 0.0f)
        {
            const float MinFirstCenter = FirstSpeechStart + FMath::Min(EventSpan * 0.42f, 0.070f);
            Center = FMath::Max(Center, MinFirstCenter);
        }

        // Boundary anchoring: the phone aligner is intentionally lightweight and
        // can place the first phone of a new phrase inside the previous acoustic
        // span when active speech time is compressed.  Do not let text-boundary
        // metadata cross a real detector gap.  This is still runtime-local
        // forced alignment: detector islands provide acoustic pause evidence, but
        // they do not own words or reorder events.
        if (EventIndex > 0 && InOutTrack.Events.Num() > 0)
        {
            const FOffgridAIAlignedVisemeEvent& PrevCommitted = InOutTrack.Events.Last();
            const bool bSentenceBoundary = T.SentenceIslandIndex != INDEX_NONE
                && PrevCommitted.SentenceIndex != INDEX_NONE
                && T.SentenceIslandIndex != PrevCommitted.SentenceIndex;
            const bool bPhraseBoundary = !bSentenceBoundary
                && T.PhraseIndex != INDEX_NONE
                && PrevCommitted.PhraseIndex != INDEX_NONE
                && T.PhraseIndex != PrevCommitted.PhraseIndex;

            if (bSentenceBoundary || bPhraseBoundary)
            {
                const TCHAR BoundaryPunctuation = BoundaryPunctuationBeforeEvent(Plan, EventIndex);
                const bool bSoftCommaBoundary = bPhraseBoundary && IsSoftCommaBoundary(BoundaryPunctuation);
                // Soft comma carry: do not convert ordinary short comma/list
                // gaps into hard phrase-start resets. A comma boundary only gets
                // a hard onset guard when the audio contains a clearly long
                // silence before the next item.
                const float RequiredGap = bSentenceBoundary ? 0.120f : (bSoftCommaBoundary ? 0.340f : 0.115f);
                const float MaxBoundarySearch = bSentenceBoundary ? 1.450f : (bSoftCommaBoundary ? 0.700f : 0.950f);
                float GapStart = -1.0f;
                float NextSpeechStart = -1.0f;
                const bool bFoundFeatureBoundary = FindNextFeatureSpeechOnsetAfter(
                    Input.AudioFeatureFrames,
                    PrevCommitted.FinalRenderCenterSeconds,
                    RequiredGap,
                    MaxBoundarySearch,
                    GapStart,
                    NextSpeechStart);

                if (!bFoundFeatureBoundary)
                {
                    const float IslandRequiredGap = bSentenceBoundary ? 0.120f : (bSoftCommaBoundary ? 0.360f : 0.185f);
                    if (FindNextSpeechIslandStartAfter(Input.SpeechIslands, PrevCommitted.FinalRenderCenterSeconds, IslandRequiredGap, NextSpeechStart))
                    {
                        GapStart = PrevCommitted.FinalRenderCenterSeconds + IslandRequiredGap;
                    }
                }

                if (NextSpeechStart >= 0.0f)
                {
                    const float BoundaryLead = bSentenceBoundary ? 0.060f : 0.045f;
                    BoundaryGuardMinCenter = NextSpeechStart + FMath::Min(EventSpan * 0.38f, BoundaryLead);
                    BoundaryGuardReason = bSentenceBoundary
                        ? FName(TEXT("sentence_boundary_onset_guard"))
                        : FName(TEXT("phrase_boundary_onset_guard"));
                    Center = FMath::Max(Center, BoundaryGuardMinCenter);
                }
                else if (!Input.bInputStreamClosed && bSentenceBoundary)
                {
                    // For terminal punctuation, avoid pre-committing the next sentence
                    // until evidence for its speech span has actually arrived.
                    break;
                }
            }
        }

        // If this is the final visible event before a text boundary,
        // keep it inside the current acoustic phrase.  The global active-speech
        // aligner can otherwise stretch a final consonant/vowel into the silence
        // before the next sentence, which reads as an isolated mouth pop.
        if (Plan.Events.IsValidIndex(EventIndex + 1))
        {
            const FOffgridAITextVisemeEvent& NextTextEvent = Plan.Events[EventIndex + 1];
            const bool bNextSentenceBoundary = NextTextEvent.SentenceIslandIndex != INDEX_NONE
                && T.SentenceIslandIndex != INDEX_NONE
                && NextTextEvent.SentenceIslandIndex != T.SentenceIslandIndex;
            const bool bNextPhraseBoundary = !bNextSentenceBoundary
                && NextTextEvent.PhraseIndex != INDEX_NONE
                && T.PhraseIndex != INDEX_NONE
                && NextTextEvent.PhraseIndex != T.PhraseIndex;
            if (bNextSentenceBoundary || bNextPhraseBoundary)
            {
                const TCHAR NextBoundaryPunctuation = BoundaryPunctuationBeforeEvent(Plan, EventIndex + 1);
                const bool bNextSoftCommaBoundary = bNextPhraseBoundary && IsSoftCommaBoundary(NextBoundaryPunctuation);
                // Soft comma carry: keep phrase-tail protection for real
                // pauses, but do not clamp a list item's tail before every short
                // comma dip. This lets the forced path keep articulation
                // continuous through supported comma gaps.
                const float RequiredTailGap = bNextSentenceBoundary ? 0.115f : (bNextSoftCommaBoundary ? 0.340f : 0.135f);
                const float MaxTailSearch = bNextSentenceBoundary ? 1.350f : (bNextSoftCommaBoundary ? 0.700f : 0.750f);
                float TailGapStart = -1.0f;
                float TailNextSpeechStart = -1.0f;
                if (FindNextFeatureSpeechOnsetAfter(Input.AudioFeatureFrames, LastCenter, RequiredTailGap, MaxTailSearch, TailGapStart, TailNextSpeechStart)
                    && TailGapStart > LastCenter + 0.045f)
                {
                    // Phrase-tail ownership: this event belongs to the current
                    // text phrase/sentence. If a real acoustic gap and next speech
                    // onset have been observed before the next text event, do not
                    // let the forced alignment path pull the current phrase
                    // tail into the next phrase.
                    TailGuardMaxCenter = TailGapStart - FMath::Min(EventSpan * 0.38f, 0.055f);
                    TailGuardMaxCenter = FMath::Max(LastCenter + 0.050f, TailGuardMaxCenter);
                    TailGuardReason = bNextSentenceBoundary
                        ? FName(TEXT("sentence_boundary_tail_guard"))
                        : FName(TEXT("phrase_boundary_tail_guard"));
                    if (Center > TailGuardMaxCenter)
                    {
                        Center = TailGuardMaxCenter;
                        PlacementReason = TailGuardReason;
                    }
                }
            }
        }

        // Streaming forced alignment owns timing.  The transcript-derived phone
        // path is segmented over observed speech frames by the aligner; runtime
        // guards only preserve phrase ownership and monotonic commit safety.
        const bool bUsedForcedAlignment = bHasPhoneEvidence;

        if (!bUsedForcedAlignment && !Input.bInputStreamClosed && !Input.bPlaybackFinalized)
        {
            // Streaming-safe: wait until the online Viterbi path has enough
            // stable audio evidence to expose this phone.
            break;
        }

        if (!bUsedForcedAlignment && (Input.bInputStreamClosed || Input.bPlaybackFinalized))
        {
            // End-of-line drain: only after the stream is closed, use the
            // transcript duration model so weak/under-resolved audio cannot
            // permanently suppress planned visemes.
            PlacementReason = FName(TEXT("final_duration_drain"));
        }

        // Phrase-start guard: once a real acoustic gap/onset has been observed
        // at a text boundary, later placement stages may not pull the first event
        // of the new phrase/sentence backward into the previous phrase tail. This
        // runs after forced-alignment placement because the aligner can place
        // a boundary phone before the acoustic phrase onset.
        if (BoundaryGuardMinCenter >= 0.0f && Center < BoundaryGuardMinCenter - 0.010f)
        {
            Center = BoundaryGuardMinCenter;
            if (!BoundaryGuardReason.IsNone())
            {
                PlacementReason = BoundaryGuardReason;
            }
        }

        // Phrase-tail guard: later placement stages may not pull the final event
        // of a phrase/sentence across a detected pause into the following phrase.
        if (TailGuardMaxCenter >= 0.0f && Center > TailGuardMaxCenter + 0.010f)
        {
            Center = TailGuardMaxCenter;
            if (!TailGuardReason.IsNone())
            {
                PlacementReason = TailGuardReason;
            }
        }

        if (Center > CommitTimeHorizon + 0.001f)
        {
            break;
        }

        const float MinOrderedCenter = LastCenter + 0.050f;
        Center = FMath::Max(Center, MinOrderedCenter);

        if (!bUsedForcedAlignment && SameSentenceAsLast(InOutTrack, T))
        {
            const float MaxGap = MaxWallClockGapForPose(Pose);
            const float MaxSameSentenceCenter = LastCenter + MaxGap;
            if (Center > MaxSameSentenceCenter)
            {
                // This guard used to always pull later states forward.  That
                // avoided long frozen holds, but on slow/list delivery it also
                // compressed valid phones into the start of a spoken phrase,
                // creating the perceptual defect of "lost" animations while
                // speech continued.  Only apply the wall-clock cap when the gap
                // between events is acoustically a pause/non-speech region.  If
                // sustained speech evidence exists between the two centers, keep
                // the phone's aligned absolute time and let the performer bridge
                // the spoken phrase with real states.
                const bool bSpeechContinues = HasSustainedRuntimeSpeechBetween(
                    Input.AudioFeatureFrames,
                    LastCenter,
                    Center,
                    0.110f);
                const bool bLongSilence = HasLongRuntimeSilenceBetween(
                    Input.AudioFeatureFrames,
                    LastCenter,
                    Center,
                    0.140f);
                if (!bSpeechContinues || bLongSilence)
                {
                    Center = FMath::Max(MinOrderedCenter, MaxSameSentenceCenter);
                }
            }
        }

        if (Center < Input.CurrentPlaybackSec + 0.005f)
        {
            Center = FMath::Max(Input.CurrentPlaybackSec + 0.025f, LastCenter + 0.050f);
        }

        FOffgridAIAlignedVisemeEvent E;
        E.EventIndex = EventIndex;
        E.PoseID = Pose;
        E.Strength = StrengthForEvent(T, Pose);
        E.SourceWord = T.SourceText;
        E.WordIndex = T.WordIndex;
        E.PhraseIndex = T.PhraseIndex;
        E.bIsStrongVisibleEvent = T.bIsStrongVisibleEvent;
        E.TextCenterNorm = EventCenterNorm(T);
        E.TextDiagnosticCenterSeconds = E.TextCenterNorm * FMath::Max(Plan.EstimatedDurationSeconds, 0.001f);
        E.FinalRenderCenterSeconds = Center;
        E.RenderStartSeconds = FMath::Max(0.0f, Center - EventSpan * 0.5f);
        E.RenderEndSeconds = Center + EventSpan * 0.5f;
        E.SourcePhoneIndex = PhoneIndex;
        const EOffgridAIPhoneClass PhoneClass = Plan.ExpectedPhones.IsValidIndex(PhoneIndex)
            ? FOffgridAIOnlinePhoneAligner::ClassForPhoneBase(Plan.ExpectedPhones[PhoneIndex].BasePhone)
            : EOffgridAIPhoneClass::Unknown;
        E.SourcePhoneClass = FName(*FOffgridAIOnlinePhoneAligner::PhoneClassToString(PhoneClass));
        E.AlignedPhoneStartSeconds = Alignment.PhoneStartSeconds.IsValidIndex(PhoneIndex) ? Alignment.PhoneStartSeconds[PhoneIndex] : 0.0f;
        E.AlignedPhoneEndSeconds = Alignment.PhoneEndSeconds.IsValidIndex(PhoneIndex) ? Alignment.PhoneEndSeconds[PhoneIndex] : 0.0f;
        E.AlignmentConfidence = Alignment.PhoneMatchScores.IsValidIndex(PhoneIndex) ? Alignment.PhoneMatchScores[PhoneIndex] : 0.0f;
        E.AlignmentReason = bHasPhoneEvidence
            ? (Alignment.PhoneAdvanceReasons.IsValidIndex(PhoneIndex) && !Alignment.PhoneAdvanceReasons[PhoneIndex].IsNone()
                ? Alignment.PhoneAdvanceReasons[PhoneIndex]
                : FName(TEXT("streaming_forced_alignment")))
            : FName(TEXT("no_phone_evidence"));
        E.CommitPlaybackSeconds = Input.CurrentPlaybackSec;
        E.CommitLeadSeconds = Center - Input.CurrentPlaybackSec;
        E.CommitReason = PlacementReason;
        E.RequiredActiveElapsedSeconds = static_cast<float>(EventIndex + 1);
        E.ObservedActiveElapsedSeconds = bUsedForcedAlignment
            ? static_cast<float>(FMath::Max(Alignment.HighestAlignedPhoneIndex + 1, 0))
            : 0.0f;
        E.ActiveProgressDeficitSeconds = FMath::Max(0.0f, E.RequiredActiveElapsedSeconds - E.ObservedActiveElapsedSeconds);
        E.RequiredProgressNorm = E.RequiredActiveElapsedSeconds;
        E.ObservedProgressNorm = E.ObservedActiveElapsedSeconds;
        E.ActiveProgressRatio = E.RequiredActiveElapsedSeconds > 0.001f ? E.ObservedActiveElapsedSeconds / E.RequiredActiveElapsedSeconds : 1.0f;
        E.bMappedToObservedSpeech = bUsedForcedAlignment;
        E.SentenceIndex = T.SentenceIslandIndex;

        InOutTrack.Events.Add(E);
        LastCenter = Center;
    }

    if (InOutTrack.Events.Num() > 0)
    {
        InOutTrack.SpeechStartSeconds = FirstSpeechStart >= 0.0f ? FirstSpeechStart : InOutTrack.Events[0].RenderStartSeconds;
        InOutTrack.SpeechEndSeconds = InOutTrack.Events.Last().RenderEndSeconds;
    }
    bInOutTrackBuilt = InOutTrack.Events.Num() > 0 || Input.bInputStreamClosed || Input.bPlaybackFinalized;
}
