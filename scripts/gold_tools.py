import csv
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import wave
from array import array
from functools import lru_cache


ROOT = pathlib.Path(__file__).resolve().parents[1]


def repo_root() -> pathlib.Path:
    return ROOT


def find_runner() -> pathlib.Path:
    bin_name = "liplab_runner.exe" if sys.platform.startswith("win") else "liplab_runner"
    candidates = [
        ROOT / "build-ninja" / bin_name,
        ROOT / "build" / bin_name,
        ROOT / "build" / "Release" / bin_name,
        ROOT / "build" / "Debug" / bin_name,
    ]
    exe = next((path for path in candidates if path.exists()), None)
    if not exe:
        raise SystemExit("liplab_runner not built; run cmake first")
    return exe


def run_runner(buffer_ms: int, chunk_ms: int) -> int:
    exe = find_runner()
    cmd = [
        str(exe),
        str(ROOT),
        "--buffer-ms",
        str(buffer_ms),
        "--chunk-ms",
        str(chunk_ms),
    ]
    return subprocess.call(cmd)


def run_runner_and_capture(buffer_ms: int, chunk_ms: int, capture_root: pathlib.Path) -> int:
    rc = run_runner(buffer_ms, chunk_ms)
    if rc != 0:
        return rc

    latest = ROOT / "outputs" / "runs" / "latest"
    if capture_root.exists():
        shutil.rmtree(capture_root)
    capture_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(latest, capture_root)
    return 0


def find_mfa_env() -> pathlib.Path:
    env = ROOT / ".conda" / "mfa"
    if not env.exists():
        raise SystemExit("MFA conda environment not found at .conda/mfa")
    return env


def mfa_root_dir() -> pathlib.Path:
    return ROOT / ".mfa"


def mfa_dictionary_path() -> pathlib.Path:
    path = mfa_root_dir() / "pretrained_models" / "dictionary" / "english_us_arpa.dict"
    if not path.exists():
        raise SystemExit(f"MFA dictionary not found: {path}")
    return path


def mfa_acoustic_model_path() -> pathlib.Path:
    path = mfa_root_dir() / "pretrained_models" / "acoustic" / "english_us_arpa.zip"
    if not path.exists():
        raise SystemExit(f"MFA acoustic model not found: {path}")
    return path


def mfa_corpus_root() -> pathlib.Path:
    return ROOT / "outputs" / "mfa_corpus" / "latest"


def mfa_align_root() -> pathlib.Path:
    return ROOT / "outputs" / "mfa_align" / "latest"


def mfa_generated_dictionary_path() -> pathlib.Path:
    return ROOT / "outputs" / "mfa_corpus" / "latest_dictionary.dict"


def mfa_generated_dictionary_report_path() -> pathlib.Path:
    return ROOT / "outputs" / "mfa_corpus" / "latest_dictionary_report.json"


def offgrid_cmudict_source_path() -> pathlib.Path:
    path = ROOT / "offgrid_dropin" / "Private" / "Lipsync" / "OffgridAICmudictData.cpp"
    if not path.exists():
        raise SystemExit(f"Offgrid CMU dictionary source not found: {path}")
    return path


def stage_mfa_corpus(cases: list[str], corpus_root: pathlib.Path) -> None:
    if corpus_root.exists():
        shutil.rmtree(corpus_root)
    corpus_root.mkdir(parents=True, exist_ok=True)
    for case_id in cases:
        transcript = read_text(ROOT / "inputs" / "transcripts" / f"{case_id}.txt")
        (corpus_root / f"{case_id}.lab").write_text(
            normalize_spoken_numbers(transcript) + "\n",
            encoding="utf-8",
        )


def normalize_dictionary_word(word: str) -> str:
    word = (word or "").strip().lower()
    word = re.sub(r"\(\d+\)$", "", word)
    return word


def transcript_word_candidates(text: str) -> list[str]:
    return re.findall(
        r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*",
        normalize_spoken_numbers(text).lower(),
    )


_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_DECADES = ["", "tens", "twenties", "thirties", "forties", "fifties", "sixties", "seventies", "eighties", "nineties"]


def _under_one_thousand_words(value: int) -> list[str]:
    words: list[str] = []
    if value >= 100:
        words.extend((_ONES[value // 100], "hundred"))
        value %= 100
    if value >= 20:
        words.append(_TENS[value // 10])
        if value % 10:
            words.append(_ONES[value % 10])
    elif value:
        words.append(_ONES[value])
    return words


def _spoken_number_words(token: str) -> list[str] | None:
    compact = token.replace(",", "").lower()
    decade = re.fullmatch(r"(\d{4})s", compact)
    if decade:
        value = int(decade.group(1))
        if value >= 1000 and value % 10 == 0:
            return _under_one_thousand_words(value // 100) + [_DECADES[(value % 100) // 10]]
    if not compact.isdigit() or len(compact) > 9:
        return None
    value = int(compact)
    if value > 999_999:
        return None
    if value == 0:
        return ["zero"]
    words: list[str] = []
    if value >= 1000:
        words.extend(_under_one_thousand_words(value // 1000))
        words.append("thousand")
        value %= 1000
    words.extend(_under_one_thousand_words(value))
    return words


def normalize_spoken_numbers(text: str) -> str:
    """Expand supported numeric orthography to the words spoken by the TTS."""

    pattern = re.compile(r"(?<![A-Za-z0-9])(?:\d{1,3}(?:,\d{3})+|\d+s|\d+)(?![A-Za-z0-9])", re.IGNORECASE)

    def replace(match: re.Match[str]) -> str:
        words = _spoken_number_words(match.group(0))
        return " ".join(words) if words else match.group(0)

    return pattern.sub(replace, text)


@lru_cache(maxsize=1)
def load_stock_dictionary_entries() -> dict[str, list[str]]:
    entries: dict[str, list[str]] = {}
    for raw_line in mfa_dictionary_path().read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        word = normalize_dictionary_word(parts[0])
        # MFA's packaged English dictionary mixes plain ``word phones...``
        # rows with rows carrying four pronunciation/silence probability
        # columns.  Those probabilities describe the source dictionary; they
        # are not phones and cannot be spliced into a derived pronunciation
        # (for example ``vanilla`` + possessive ``s``).
        phone_parts = parts[1:]
        while phone_parts:
            try:
                float(phone_parts[0])
            except ValueError:
                break
            phone_parts.pop(0)
        if not phone_parts:
            continue
        pronunciation = " ".join(phone_parts)
        entries.setdefault(word, [])
        if pronunciation not in entries[word]:
            entries[word].append(pronunciation)
    return entries


@lru_cache(maxsize=1)
def load_offgrid_dictionary_entries() -> dict[str, list[str]]:
    entries: dict[str, list[str]] = {}
    pattern = re.compile(r'R"CMU\((.*?)\)CMU"', re.DOTALL)
    text = offgrid_cmudict_source_path().read_text(encoding="utf-8")
    for block in pattern.findall(text):
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            word = normalize_dictionary_word(parts[0])
            pronunciation = " ".join(parts[1:])
            entries.setdefault(word, [])
            if pronunciation not in entries[word]:
                entries[word].append(pronunciation)
    return entries


def combine_dictionary_entries() -> dict[str, list[str]]:
    combined: dict[str, list[str]] = {}
    for source in (load_stock_dictionary_entries(), load_offgrid_dictionary_entries()):
        for word, pronunciations in source.items():
            combined.setdefault(word, [])
            for pronunciation in pronunciations:
                if pronunciation not in combined[word]:
                    combined[word].append(pronunciation)
    return combined


def build_compound_pronunciation(word: str, entries: dict[str, list[str]]) -> str | None:
    direct_parts = [part for part in re.split(r"[-']", word) if part]
    if len(direct_parts) > 1 and all(part in entries for part in direct_parts):
        phones: list[str] = []
        for part in direct_parts:
            phones.extend(entries[part][0].split())
        return " ".join(phones)

    for split_index in range(3, len(word) - 2):
        left = word[:split_index]
        right = word[split_index:]
        if left in entries and right in entries:
            return f"{entries[left][0]} {entries[right][0]}"

    if word.endswith("s") and len(word) > 3:
        singular = word[:-1]
        if singular in entries:
            return f"{entries[singular][0]} Z"
    if word.endswith("es") and len(word) > 4:
        singular = word[:-2]
        if singular in entries:
            return f"{entries[singular][0]} Z"
    if word.endswith("ed") and len(word) > 4:
        stem = word[:-2]
        if stem in entries:
            return f"{entries[stem][0]} D"
    if word.endswith("ing") and len(word) > 5:
        stem = word[:-3]
        if stem in entries:
            return f"{entries[stem][0]} IH0 NG"
    return None



_DIGIT_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}

_LETTER_WORDS = {
    "a": "EY1", "b": "B IY1", "c": "S IY1", "d": "D IY1", "e": "IY1", "f": "EH1 F",
    "g": "JH IY1", "h": "EY1 CH", "i": "AY1", "j": "JH EY1", "k": "K EY1", "l": "EH1 L",
    "m": "EH1 M", "n": "EH1 N", "o": "OW1", "p": "P IY1", "q": "K Y UW1", "r": "AA1 R",
    "s": "EH1 S", "t": "T IY1", "u": "Y UW1", "v": "V IY1", "w": "D AH1 B AH0 L Y UW0",
    "x": "EH1 K S", "y": "W AY1", "z": "Z IY1",
}

_MULTI_GRAPHEME_PHONES = [
    ("ough", "AO1"), ("eigh", "EY1"), ("tion", "SH AH0 N"), ("sion", "ZH AH0 N"),
    ("ch", "CH"), ("sh", "SH"), ("th", "TH"), ("ph", "F"), ("wh", "W"),
    ("ng", "NG"), ("qu", "K W"), ("ck", "K"), ("ee", "IY1"), ("ea", "IY1"),
    ("ai", "EY1"), ("ay", "EY1"), ("oa", "OW1"), ("ow", "AW1"), ("oo", "UW1"),
    ("ou", "AW1"), ("oi", "OY1"), ("oy", "OY1"), ("er", "ER0"), ("ar", "AA1 R"),
    ("or", "AO1 R"), ("ir", "ER1"), ("ur", "ER1"),
]

_SINGLE_GRAPHEME_PHONES = {
    "a": "AH0", "b": "B", "c": "K", "d": "D", "e": "EH1", "f": "F", "g": "G", "h": "HH",
    "i": "IH1", "j": "JH", "k": "K", "l": "L", "m": "M", "n": "N", "o": "OW1", "p": "P",
    "q": "K", "r": "R", "s": "S", "t": "T", "u": "AH0", "v": "V", "w": "W", "x": "K S",
    "y": "IY0", "z": "Z",
}


def _entry_pronunciation_for_word(word: str, entries: dict[str, list[str]]) -> str | None:
    pronunciations = entries.get(normalize_dictionary_word(word))
    return pronunciations[0] if pronunciations else None


def build_number_pronunciation(word: str, entries: dict[str, list[str]]) -> str | None:
    """Return a dictionary-backed pronunciation for numeric tokens.

    This deliberately uses digit-by-digit expansion because the corpus contains
    IDs / codes like 000 or 4, where reading the token as a number can be less
    useful than preserving the spoken sequence that MFA is likely hearing.
    """
    if not re.fullmatch(r"\d+", word or ""):
        return None
    phones: list[str] = []
    for digit in word:
        digit_word = _DIGIT_WORDS.get(digit)
        pronunciation = _entry_pronunciation_for_word(digit_word or "", entries)
        if not pronunciation:
            return None
        phones.extend(pronunciation.split())
    return " ".join(phones)


def build_acronym_pronunciation(word: str) -> str | None:
    letters = re.sub(r"[^a-z]", "", word.lower())
    if not letters or letters != word.lower() or len(letters) > 5:
        return None
    # Very short all-consonant tokens are often letter names rather than words.
    if any(ch in "aeiou" for ch in letters):
        return None
    phones: list[str] = []
    for ch in letters:
        pronunciation = _LETTER_WORDS.get(ch)
        if not pronunciation:
            return None
        phones.extend(pronunciation.split())
    return " ".join(phones)


def build_grapheme_fallback_pronunciation(word: str) -> str | None:
    """Small deterministic ARPAbet fallback for proper names / nonce words.

    This is not intended to be as good as a real G2P model. Its purpose is to
    keep MFA from collapsing unknown spoken words to spn, so the resulting gold
    still has editable phone intervals instead of a single opaque unknown span.
    """
    letters = re.sub(r"[^a-z]", "", word.lower())
    if len(letters) < 2:
        return None
    phones: list[str] = []
    i = 0
    while i < len(letters):
        matched = False
        for grapheme, pronunciation in _MULTI_GRAPHEME_PHONES:
            if letters.startswith(grapheme, i):
                phones.extend(pronunciation.split())
                i += len(grapheme)
                matched = True
                break
        if matched:
            continue
        ch = letters[i]
        pronunciation = _SINGLE_GRAPHEME_PHONES.get(ch)
        if pronunciation:
            # Soft c/g before front vowels.
            if ch == "c" and i + 1 < len(letters) and letters[i + 1] in "eiy":
                pronunciation = "S"
            elif ch == "g" and i + 1 < len(letters) and letters[i + 1] in "eiy":
                pronunciation = "JH"
            phones.extend(pronunciation.split())
        i += 1
    # Avoid a totally consonant-only pronunciation for name-like words.
    if not any(any(ch.isdigit() for ch in phone) or phone in {"AH", "EH", "IH", "IY", "OW", "UW", "AA", "AO", "AE", "AY", "EY", "OY", "AW", "ER"} for phone in phones):
        phones.append("AH0")
    return " ".join(phones) if phones else None


def build_unknown_word_pronunciation(word: str, entries: dict[str, list[str]]) -> tuple[str | None, str | None]:
    number = build_number_pronunciation(word, entries)
    if number:
        return number, "number_digit_fallback"
    acronym = build_acronym_pronunciation(word)
    if acronym:
        return acronym, "acronym_letter_fallback"
    guessed = build_grapheme_fallback_pronunciation(word)
    if guessed:
        return guessed, "grapheme_g2p_fallback"
    return None, None


def build_mfa_alignment_dictionary(cases: list[str], output_path: pathlib.Path) -> dict[str, object]:
    entries = combine_dictionary_entries()
    selected: dict[str, list[str]] = {}
    unresolved: list[str] = []
    resolved_by_source = {
        "stock_or_offgrid": 0,
        "compound_fallback": 0,
        "number_digit_fallback": 0,
        "acronym_letter_fallback": 0,
        "grapheme_g2p_fallback": 0,
    }
    fallback_pronunciations: dict[str, dict[str, str]] = {}

    for case_id in cases:
        transcript = read_text(ROOT / "inputs" / "transcripts" / f"{case_id}.txt")
        for token in transcript_word_candidates(transcript):
            word = normalize_dictionary_word(token)
            if not word or word in selected:
                continue
            pronunciations = entries.get(word)
            if pronunciations:
                selected[word] = pronunciations
                resolved_by_source["stock_or_offgrid"] += 1
                continue
            compound = build_compound_pronunciation(word, entries)
            if compound:
                selected[word] = [compound]
                resolved_by_source["compound_fallback"] += 1
                continue
            guessed, source = build_unknown_word_pronunciation(word, entries)
            if guessed and source:
                selected[word] = [guessed]
                resolved_by_source[source] += 1
                fallback_pronunciations[word] = {"source": source, "pronunciation": guessed}
                continue
            unresolved.append(word)

    lines: list[str] = []
    for word in sorted(selected):
        for pronunciation in selected[word]:
            lines.append(f"{word}\t{pronunciation}")
    write_text(output_path, "\n".join(lines) + "\n")

    report = {
        "case_count": len(cases),
        "word_count": len(selected),
        "resolved_by_source": resolved_by_source,
        "fallback_pronunciations": fallback_pronunciations,
        "unresolved_words": sorted(set(unresolved)),
    }
    write_json(mfa_generated_dictionary_report_path(), report)
    return report


def run_mfa_align(cases: list[str], output_root: pathlib.Path, num_jobs: int) -> int:
    env_path = find_mfa_env()
    corpus_root = mfa_corpus_root()
    stage_mfa_corpus(cases, corpus_root)
    generated_dictionary = mfa_generated_dictionary_path()
    build_mfa_alignment_dictionary(cases, generated_dictionary)

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["MFA_ROOT_DIR"] = str(mfa_root_dir())
    env["CONDA_NO_PLUGINS"] = "true"
    path_entries = [
        str(env_path / "Library" / "bin"),
        str(env_path / "Scripts"),
        str(env_path / "bin"),
        env.get("PATH", ""),
    ]
    env["PATH"] = os.pathsep.join([entry for entry in path_entries if entry])

    # Prefer the project-local MFA executable directly. This avoids depending on
    # conda.exe being on PATH when the user already activated .conda/mfa.
    if os.name == "nt":
        mfa_exe = env_path / "Scripts" / "mfa.exe"
    else:
        mfa_exe = env_path / "bin" / "mfa"

    if mfa_exe.exists():
        cmd = [
            str(mfa_exe),
            "align",
            str(corpus_root),
            str(generated_dictionary),
            str(mfa_acoustic_model_path()),
            str(output_root),
            "--audio_directory",
            str(ROOT / "inputs" / "wav"),
            "--clean",
            "--single_speaker",
            "--num_jobs",
            str(max(num_jobs, 1)),
        ]
    else:
        # Fallback for environments where only conda-run is available.
        cmd = [
            "conda",
            "run",
            "-p",
            str(env_path),
            "mfa",
            "align",
            str(corpus_root),
            str(generated_dictionary),
            str(mfa_acoustic_model_path()),
            str(output_root),
            "--audio_directory",
            str(ROOT / "inputs" / "wav"),
            "--clean",
            "--single_speaker",
            "--num_jobs",
            str(max(num_jobs, 1)),
        ]
    print("Running MFA:", " ".join(cmd))
    return subprocess.call(cmd, env=env)


def case_stems() -> list[str]:
    with (ROOT / "inputs" / "corpus.csv").open(newline="", encoding="utf-8") as handle:
        return sorted(row["case_id"] for row in csv.DictReader(handle))


def read_text(path: pathlib.Path) -> str:
    # Several imported Offgrid transcripts carry a UTF-8 BOM.  Treat it as an
    # encoding marker, not as part of the first word handed to MFA.
    return path.read_text(encoding="utf-8-sig").strip()


def read_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv_rows(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def wav_metadata(path: pathlib.Path) -> dict[str, float | int | str]:
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        frame_count = wav_file.getnframes()
        duration_sec = frame_count / sample_rate if sample_rate > 0 else 0.0
        return {
            "path": path.relative_to(ROOT).as_posix(),
            "sample_rate_hz": sample_rate,
            "channels": channels,
            "frame_count": frame_count,
            "duration_sec": duration_sec,
        }


def read_wav_mono_samples(path: pathlib.Path) -> tuple[int, list[float]]:
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_count = wav_file.getnframes()
        if sample_width != 2:
            raise SystemExit(f"Only PCM16 WAV is supported: {path}")

        raw = wav_file.readframes(frame_count)
        pcm = array("h")
        pcm.frombytes(raw)

    if channels <= 0:
        return sample_rate, []

    if channels == 1:
        return sample_rate, [sample / 32768.0 for sample in pcm]

    mono: list[float] = []
    for index in range(0, len(pcm), channels):
        frame = pcm[index : index + channels]
        mono.append(sum(frame) / (32768.0 * len(frame)))
    return sample_rate, mono


def latest_run_case_dir(case_id: str) -> pathlib.Path:
    return ROOT / "outputs" / "runs" / "latest" / case_id


def offline_gold_run_root() -> pathlib.Path:
    return ROOT / "outputs" / "offline_gold" / "latest"


def offline_gold_case_dir(case_id: str) -> pathlib.Path:
    return offline_gold_run_root() / case_id


def gold_draft_root() -> pathlib.Path:
    return ROOT / "outputs" / "gold_drafts"


def gold_draft_dir(case_id: str) -> pathlib.Path:
    return gold_draft_root() / case_id


def gold_inputs_root() -> pathlib.Path:
    return ROOT / "inputs" / "gold"


def approved_gold_case_dir(case_id: str) -> pathlib.Path:
    return gold_inputs_root() / case_id


def gold_manifest_path() -> pathlib.Path:
    return gold_inputs_root() / "manifest.json"


def mfa_textgrid_path(case_id: str) -> pathlib.Path:
    return mfa_align_root() / f"{case_id}.TextGrid"


def parse_float(value: str | float | int | None, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: str | float | int | None, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def strip_phone_stress(phone: str) -> str:
    return re.sub(r"\d", "", (phone or "").strip().upper())


def normalize_word(word: str) -> str:
    return re.sub(r"[^a-z0-9']+", "", (word or "").strip().lower())


def parse_long_textgrid(path: pathlib.Path) -> dict[str, list[dict[str, object]]]:
    if not path.exists():
        raise SystemExit(f"MFA TextGrid missing: {path}")

    tier_name: str | None = None
    current: dict[str, object] | None = None
    tiers: dict[str, list[dict[str, object]]] = {"words": [], "phones": []}

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith('name = "'):
            tier_name = line[len('name = "') : -1]
            continue
        if line.startswith("intervals ["):
            current = {"xmin": None, "xmax": None, "text": ""}
            continue
        if current is None or tier_name not in tiers:
            continue
        if line.startswith("xmin = "):
            current["xmin"] = float(line[len("xmin = ") :].strip())
        elif line.startswith("xmax = "):
            current["xmax"] = float(line[len("xmax = ") :].strip())
        elif line.startswith('text = "'):
            current["text"] = line[len('text = "') : -1]
            tiers[tier_name].append(
                {
                    "start": float(current["xmin"] or 0.0),
                    "end": float(current["xmax"] or 0.0),
                    "text": str(current["text"]),
                }
            )
            current = None

    return tiers
