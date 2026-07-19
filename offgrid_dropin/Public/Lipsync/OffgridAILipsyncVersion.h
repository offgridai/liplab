#pragma once

#include "CoreMinimal.h"

// Bump this whenever the authoritative shared lipsync behavior changes. The
// value is emitted by every host's diagnostics so logs can be tied to the
// exact transplant contract that produced them.
#define OFFGRIDAI_LIPSYNC_IMPLEMENTATION_VERSION TEXT("2026.07.19-boundary-final-syllable-v7")

// Version of the focused diagnostic files written by host integrations.
#define OFFGRIDAI_LIPSYNC_DIAGNOSTIC_SCHEMA_VERSION 5
