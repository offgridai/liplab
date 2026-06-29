# extract_liplab_inputs_from_offgrid_logs.py
#
# Extracts OffgridAI LipsyncDebug TTS WAVs + spoken transcript text into liplab inputs.
#
# Source folders:
#   C:\Users\matthewpriestley_kil\Documents\Unreal Projects\OffgridAIMetahuman\Saved\Logs\OffgridAI\LipsyncDebug\<per-line-folder>\
#
# Output:
#   C:\git\liplab\inputs\wav\case_000100.wav
#   C:\git\liplab\inputs\transcripts\case_000100.txt
#
# Usage:
#   py extract_liplab_inputs_from_offgrid_logs.py
#
# Optional:
#   change START_INDEX if you want to append after existing cases.

from pathlib import Path
import shutil
import re
import sys

SOURCE_ROOT = Path(
    r"C:\Users\matthewpriestley_kil\Documents\Unreal Projects\OffgridAIMetahuman\Saved\Logs\OffgridAI\LipsyncDebug"
)

OUTPUT_WAV = Path(r"C:\git\liplab\inputs\wav")
OUTPUT_TXT = Path(r"C:\git\liplab\inputs\transcripts")

START_INDEX = 100
DRY_RUN = False

# Only use explicit spoken-text fields. Do NOT fall back to longest line,
# because metadata contains long CSV=/WAV= paths.
TRANSCRIPT_KEYS = [
    "Dialogue",
    "Transcript",
    "SpokenText",
    "NPCLine",
    "LineText",
    "Text",
]

BAD_PREFIXES = (
    "WAV=",
    "CSV=",
    "NPCID=",
    "LineID=",
    "ConversationID=",
    "VoiceID=",
    "Emotion=",
    "ActiveSampleRate=",
    "ActiveNumChannels=",
    "TextPlan",
    "V49",
    "V56",
    "V58",
    "V59",
    "V62",
    "V63",
)

def read_text_flexible(path: Path) -> str:
    data = path.read_bytes()

    # UTF-16 files commonly contain NUL bytes; try that first if present.
    if b"\x00" in data[:200]:
        for enc in ("utf-16", "utf-16-le", "utf-16-be"):
            try:
                return data.decode(enc).replace("\x00", "")
            except UnicodeDecodeError:
                pass

    for enc in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return data.decode(enc).replace("\x00", "")
        except UnicodeDecodeError:
            pass

    return data.decode("utf-8", errors="ignore").replace("\x00", "")

def extract_transcript(metadata_path: Path) -> str | None:
    text = read_text_flexible(metadata_path)

    # Primary format observed:
    #   Dialogue=Welcome to Alfie's Bodega. How can I help you today?
    #
    # Accept either "=" or ":".
    for key in TRANSCRIPT_KEYS:
        pattern = rf"^\s*{re.escape(key)}\s*[:=]\s*(.*?)\s*$"
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            value = match.group(1).strip().strip('"')
            if value and not value.upper().startswith(("WAV=", "CSV=")):
                return value

    # Do not use generic longest-line fallback. It misidentifies WAV= paths as transcript.
    return None

def pick_wav(folder: Path) -> Path | None:
    preferred = sorted(folder.glob("tts_line_*.wav"))
    if preferred:
        return preferred[0]

    wavs = sorted(folder.glob("*.wav"))
    if wavs:
        return wavs[0]

    return None

def next_available_index(start: int) -> int:
    idx = start
    while True:
        case = f"case_{idx:06d}"
        if not (OUTPUT_WAV / f"{case}.wav").exists() and not (OUTPUT_TXT / f"{case}.txt").exists():
            return idx
        idx += 1

def main() -> int:
    if not SOURCE_ROOT.exists():
        print(f"[ERROR] Source root does not exist: {SOURCE_ROOT}")
        return 1

    OUTPUT_WAV.mkdir(parents=True, exist_ok=True)
    OUTPUT_TXT.mkdir(parents=True, exist_ok=True)

    folders = sorted(p for p in SOURCE_ROOT.iterdir() if p.is_dir())
    if not folders:
        print(f"[ERROR] No line folders found under: {SOURCE_ROOT}")
        return 1

    case_index = next_available_index(START_INDEX)
    success_count = 0
    skip_count = 0

    for folder in folders:
        metadata = folder / "line_metadata.txt"
        wav_src = pick_wav(folder)

        if not metadata.exists():
            print(f"[SKIP] No line_metadata.txt: {folder.name}")
            skip_count += 1
            continue

        if wav_src is None:
            print(f"[SKIP] No WAV: {folder.name}")
            skip_count += 1
            continue

        transcript = extract_transcript(metadata)
        if not transcript:
            print(f"[SKIP] No explicit Dialogue/Transcript field: {folder.name}")
            skip_count += 1
            continue

        if transcript.startswith(BAD_PREFIXES) or "Saved/Logs/OffgridAI" in transcript or "\\Saved\\Logs\\OffgridAI" in transcript:
            print(f"[SKIP] Suspicious transcript '{transcript[:80]}': {folder.name}")
            skip_count += 1
            continue

        case_name = f"case_{case_index:06d}"
        wav_dst = OUTPUT_WAV / f"{case_name}.wav"
        txt_dst = OUTPUT_TXT / f"{case_name}.txt"

        print(f"[OK] {case_name}: {transcript}")

        if not DRY_RUN:
            shutil.copy2(wav_src, wav_dst)
            txt_dst.write_text(transcript.strip() + "\n", encoding="utf-8")

        case_index += 1
        success_count += 1

    print()
    print("Completed.")
    print(f"Success: {success_count}")
    print(f"Skipped: {skip_count}")
    print(f"Next available case index: {case_index}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
