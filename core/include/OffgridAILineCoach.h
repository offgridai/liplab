
#pragma once
#include "CoreMinimal.h"

struct FOffgridAIAlignedVisemeEvent
{
    int32 EventIndex = INDEX_NONE;
    FName PoseID = NAME_None;
    float Strength = 0.0f;
    FString SourceWord;
    int32 WordIndex = INDEX_NONE;
    int32 PhraseIndex = 0;
    bool bIsLandmark = false;
    float TextCenterNorm = 0.0f;
    float TextDiagnosticCenterSeconds = 0.0f;
    float FinalRenderCenterSeconds = 0.0f;
    float RenderStartSeconds = 0.0f;
    float RenderEndSeconds = 0.0f;
    float AlignmentConfidence = 0.0f;
    bool bUsedLayer1Fallback = false;
    float RawShiftSeconds = 0.0f;
    float AppliedShiftSeconds = 0.0f;
    float ShiftCapSeconds = 0.0f;

    // Alpha05 diagnostic-only invariant repair. The streaming aligner preserves
    // text event order, so committed centers must be strictly non-decreasing in
    // that same order after all island/pocket transforms have run.
    bool bCenterOrderRepaired = false;
    float CenterOrderRepairSeconds = 0.0f;

    // Step 9 audio-evidence nudge diagnostics. These are diagnostic-only and
    // describe what the local audio aligner searched for, what it found, and
    // why the candidate was accepted or rejected. Runtime sampling must continue
    // to use FinalRenderCenterSeconds as the authoritative clock.
    bool bAudioNudgeEligible = false;
    bool bAudioNudgeSearchPerformed = false;
    bool bAudioNudgeAccepted = false;
    FName AudioNudgeRejectReason = NAME_None;
    float AudioNudgeSearchStartSeconds = 0.0f;
    float AudioNudgeSearchEndSeconds = 0.0f;
    float AudioNudgeObservedEndSeconds = 0.0f;
    float AudioNudgeScheduledCenterSeconds = 0.0f;
    float AudioNudgeCandidateRawCenterSeconds = 0.0f;
    float AudioNudgeCandidateAppliedCenterSeconds = 0.0f;
    float AudioNudgeCandidateRawShiftSeconds = 0.0f;
    float AudioNudgeCandidateAppliedShiftSeconds = 0.0f;
    float AudioNudgeCandidateConfidence = 0.0f;
    float AudioNudgeRequiredConfidence = 0.0f;
    float AudioNudgeMaxCorrectionSeconds = 0.0f;

    // G06 passive streaming correspondence diagnostics. These deliberately
    // separate acoustic correspondence from 150 ms preroll actionability:
    // a landmark can be a real match even when it arrived too late to safely
    // change the already-authored runtime track.
    bool bAudioNudgeCorrespondenceMatched = false;
    bool bAudioNudgeActionableWithinPreroll = false;
    float AudioNudgeCandidateAudioTimeSeconds = 0.0f;
    float AudioNudgeCandidateAvailableAtSeconds = 0.0f;

    // G07 passive compatibility diagnostics.  These fields record the actual
    // streaming detector class that was paired with this planned viseme and
    // whether the match was exact or family-compatible.  They do not alter
    // timing; they exist so future runtime retiming can be based on measured
    // detector/planner vocabulary overlap rather than hidden class mismatches.
    FName AudioNudgeCandidateLandmarkClass = NAME_None;
    FName AudioNudgeCompatibilityKind = NAME_None;
    float AudioNudgeCompatibilityScore = 0.0f;

    // V22 diagnostics: when a candidate was not searched because the search
    // window was empty, classify whether this was truly no audio, a clipped
    // window, or the important case where usable audio existed before the
    // planner-predicted landmark center.
    FName AudioNudgeEmptyWindowCause = NAME_None;
    float AudioNudgeAvailableAudioStartSeconds = 0.0f;
    float AudioNudgeAvailableAudioEndSeconds = 0.0f;
    float AudioNudgeAvailableBeforeSearchSeconds = 0.0f;
    float AudioNudgePredictedLeadSeconds = 0.0f;

    // V23 diagnostics: identifies whether the nudge candidate used the old
    // planner-centered window or the new available-audio early-search window.
    FName AudioNudgeSearchMode = NAME_None;

    float CommitPlaybackSeconds = 0.0f;
    float CommitLeadSeconds = 0.0f;
    FName CommitReason = FName(TEXT("unknown"));
    float EffectiveCommitHorizonSeconds = 0.0f;
    float EffectiveSegmentScale = 1.0f;

    // N04 diagnostic-only active-clock accounting. These fields explain why
    // each event was placed by audio occupancy or tail drain: how much
    // speech-active audio the event required from the text plan, how much
    // observed active audio existed, and the resulting deficit. Runtime
    // sampling continues to use FinalRenderCenterSeconds as authoritative.
    float RequiredActiveElapsedSeconds = 0.0f;
    float ObservedActiveElapsedSeconds = 0.0f;
    float ActiveProgressDeficitSeconds = 0.0f;
    float RequiredProgressNorm = 0.0f;
    float ObservedProgressNorm = 0.0f;
    float ActiveProgressRatio = 1.0f;
    bool bMappedToObservedSpeech = false;

    // Island-clock diagnostics used by the current streaming sentence-island drain model.
    int32 TextIslandIndex = INDEX_NONE;
    int32 AudioIslandIndex = INDEX_NONE;
    float IslandTextStartSeconds = 0.0f;
    float IslandTextEndSeconds = 0.0f;
    float IslandAudioStartSeconds = 0.0f;
    float IslandAudioEndSeconds = 0.0f;
    float IslandLocalNorm = 0.0f;
    float IslandAudioSpanSeconds = 0.0f;
    bool bReusedIslandClock = false;

    // Explicit planner-owned timing diagnostics. These fields separate the
    // text/prosody planner clock from the runtime playback clock.
    float PlannerIslandPredictedDurationSeconds = 0.0f;
    float PlannerIslandSpeechMaterialSeconds = 0.0f;
    float PlannerIslandPunctuationSeconds = 0.0f;
    float PlannerIslandShortUtteranceFloorSeconds = 0.0f;
    int32 PlannerIslandSyllableCount = 0;
    float PlannerIslandSyllableFloorSeconds = 0.0f;
    float PlannerEventPredictedOffsetSeconds = 0.0f;
    float PlannerEventNormCenter = 0.0f;
    int32 PlannerProsodyGroupIndex = INDEX_NONE;
    FName PlannerProsodyRole = NAME_None;
    float PlannerProsodyGroupWeight = 0.0f;
    float PlannerProsodyGroupAllocatedSeconds = 0.0f;
    int32 PlannerProsodyGroupEventCount = 0;
    float PlannerProsodyGroupEventCountModelSeconds = 0.0f;
    float PlannerProsodyIslandEventCountModelSeconds = 0.0f;
    float PlannerProsodyIslandSpeechBudgetSeconds = 0.0f;
    float PlannerProsodyAllocationScale = 0.0f;

    // AU18b duration-prior diagnostics. Text/prosody still owns duration;
    // this records the corpus-fitted event-count bucket correction applied
    // before runtime launch/landmark nudging.
    FName PlannerDurationBucket = NAME_None;
    float PlannerDurationScaleApplied = 1.0f;
    float PlannerIslandUnscaledDurationSeconds = 0.0f;

};

struct FOffgridAIAlignedVisemeTrack
{
    FName NPCID = NAME_None;
    FName LineID = NAME_None;
    float SpeechStartSeconds = 0.0f;
    float SpeechEndSeconds = 0.0f;
    TArray<FOffgridAIAlignedVisemeEvent> Events;
};

struct FOffgridAIPerformedVisemeFrame
{
    float PlaybackSeconds = 0.0f;
    FName NPCID = NAME_None;
    FName LineID = NAME_None;
    TMap<FName, float> AbstractVisemeWeights;
};

struct FOffgridAISubmittedVisemeSample
{
    int32 EventIndex = INDEX_NONE;
    FName PoseID = NAME_None;
    FString SourceWord;
    float PlaybackSeconds = 0.0f;
    float CommittedRenderStartSeconds = 0.0f;
    float CommittedRenderCenterSeconds = 0.0f;
    float CommittedRenderEndSeconds = 0.0f;
    float SubmittedWeight = 0.0f;
    float SourceStrength = 0.0f;
};

