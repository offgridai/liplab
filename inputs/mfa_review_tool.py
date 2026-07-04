#!/usr/bin/env python3
"""
MFA Alignment Review Tool

Run from the dataset root that contains:
  wav/<case_id>.wav
  transcripts/<case_id>.txt
  gold/<case_id>/words.csv
  gold/<case_id>/phones.csv or visemes.csv
  gold/<case_id>/speech.csv

Normal playback remains dependency-free. Premiere-style continuous audio scrubbing uses
the optional `sounddevice` package when available; without it the tool falls back
to the older best-effort temp-WAV scrub. Install with: pip install sounddevice

Playback-rate control is implemented by writing a temporary WAV segment with
an adjusted sample rate. This is dependency-free, but it changes pitch as well
as speed. Use 0.5x/0.75x for careful review and 1.0x for normal playback.

Premiere-style scrub: clicking/dragging the playhead or empty waveform space
plays short audio slices only while the cursor is moving, including very slow steady drags. Dragging left plays
reversed slices; dragging right plays forward slices; holding still is silent; releasing stops.
"""
from __future__ import annotations

import csv
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

LAYER_FILES = ("words.csv", "visemes.csv", "speech.csv")
MARKER_COLORS = {
    "words": "#3b82f6",
    "visemes": "#dc2626",
    "speech": "#16a34a",
}


def marker_label(row: dict[str, str]) -> str:
    return (
        row.get("phone")
        or row.get("source_phone")
        or row.get("pose")
        or row.get("word")
        or row.get("index")
        or ""
    )

@dataclass
class Case:
    case_id: str
    wav_path: Path
    transcript_path: Optional[Path]
    gold_dir: Path

class WavData:
    def __init__(self, path: Path):
        self.path = path
        with wave.open(str(path), "rb") as wf:
            self.channels = wf.getnchannels()
            self.sample_width = wf.getsampwidth()
            self.rate = wf.getframerate()
            self.frames = wf.getnframes()
            self.duration = self.frames / float(self.rate) if self.rate else 0.0
            raw = wf.readframes(self.frames)
        self.raw = raw
        self.samples = self._decode_mono(raw)

    def _decode_mono(self, raw: bytes) -> List[float]:
        if self.sample_width != 2:
            raise ValueError(f"Only 16-bit PCM WAV is supported; got sample width {self.sample_width}")
        import struct
        count = len(raw) // 2
        vals = struct.unpack("<" + "h" * count, raw)
        if self.channels <= 1:
            return [v / 32768.0 for v in vals]
        out = []
        for i in range(0, len(vals), self.channels):
            out.append(sum(vals[i:i+self.channels]) / (32768.0 * self.channels))
        return out

    def write_segment(self, start_sec: float, out_path: Path, playback_rate: float = 1.0, duration_sec: Optional[float] = None) -> float:
        start_sec = max(0.0, min(start_sec, self.duration))
        playback_rate = max(0.25, min(2.0, playback_rate))
        start_frame = int(start_sec * self.rate)
        max_frames = None
        if duration_sec is not None:
            max_frames = max(1, int(max(0.001, duration_sec) * self.rate))
        with wave.open(str(self.path), "rb") as src:
            src.setpos(start_frame)
            remaining = src.getnframes() - start_frame
            frames_to_read = remaining if max_frames is None else min(remaining, max_frames)
            data = src.readframes(frames_to_read)
            params = src.getparams()
        # Dependency-free rate change: keep samples intact but change the WAV
        # frame rate in the temporary playback file. This changes pitch too,
        # but works with winsound/afplay/aplay/paplay and keeps the tool zero-install.
        params = params._replace(framerate=max(1, int(round(self.rate * playback_rate))))
        with wave.open(str(out_path), "wb") as dst:
            dst.setparams(params)
            dst.writeframes(data)
        real_audio_duration = (len(data) / max(1, params.nchannels * params.sampwidth)) / max(1, self.rate)
        return real_audio_duration / playback_rate

    def write_slice(self, start_sec: float, end_sec: float, out_path: Path, playback_rate: float = 1.0, reverse: bool = False) -> float:
        start_sec = max(0.0, min(start_sec, self.duration))
        end_sec = max(0.0, min(end_sec, self.duration))
        if end_sec < start_sec:
            start_sec, end_sec = end_sec, start_sec
            reverse = not reverse
        playback_rate = max(0.25, min(2.0, playback_rate))
        start_frame = int(start_sec * self.rate)
        end_frame = max(start_frame + 1, int(end_sec * self.rate))
        with wave.open(str(self.path), "rb") as src:
            src.setpos(start_frame)
            data = src.readframes(end_frame - start_frame)
            params = src.getparams()
        if reverse and data:
            frame_size = params.nchannels * params.sampwidth
            frames = [data[i:i+frame_size] for i in range(0, len(data), frame_size)]
            frames.reverse()
            data = b"".join(frames)
        params = params._replace(framerate=max(1, int(round(self.rate * playback_rate))))
        with wave.open(str(out_path), "wb") as dst:
            dst.setparams(params)
            dst.writeframes(data)
        real_audio_duration = (len(data) / max(1, params.nchannels * params.sampwidth)) / max(1, self.rate)
        return real_audio_duration / playback_rate


class ContinuousScrubber:
    """Callback-driven audio scrubber.

    `winsound` cannot reliably restart dozens of tiny files per second during
    mouse drag.  This class keeps one low-latency output stream open and swaps
    in the latest tiny audio slice. Each slice is consumed once; when the
    playhead stops moving the stream outputs silence instead of looping.
    """
    def __init__(self):
        self.sd = None
        self.stream = None
        self.slice_bytes = b""
        self.offset = 0
        self.frame_size = 2
        self.rate = 0
        self.channels = 1
        self.playback_rate = 1.0
        self.active = False
        self.last_error: Optional[str] = None
        self._lock = None

    def available(self) -> bool:
        try:
            import sounddevice as sd  # type: ignore
            import threading
            self.sd = sd
            if self._lock is None:
                self._lock = threading.Lock()
            return True
        except Exception as e:
            self.last_error = str(e)
            return False

    def start(self, wav: WavData, playback_rate: float = 1.0):
        if not self.available():
            raise RuntimeError("Continuous scrub requires sounddevice. Install with: pip install sounddevice")
        playback_rate = max(0.25, min(2.0, playback_rate))
        out_rate = max(1, int(round(wav.rate * playback_rate)))
        if self.stream and self.active and self.rate == out_rate and self.channels == wav.channels:
            return
        self.stop()
        self.rate = out_rate
        self.channels = wav.channels
        self.frame_size = wav.channels * wav.sample_width
        self.playback_rate = playback_rate
        self.slice_bytes = b""
        self.offset = 0

        def callback(outdata, frames, time_info, status):
            nbytes = len(outdata)
            with self._lock:
                src = self.slice_bytes
                if not src:
                    outdata[:] = b"\x00" * nbytes
                    return
                # Play the current scrub slice once. Do not loop it: when the
                # mouse/playhead is stationary, Premiere-style scrub should fall
                # silent rather than repeating the held clip.
                remaining = max(0, len(src) - self.offset)
                take = min(nbytes, remaining)
                if take > 0:
                    out = bytearray(src[self.offset:self.offset + take])
                    self.offset += take
                else:
                    out = bytearray()
                if self.offset >= len(src):
                    self.slice_bytes = b""
                    self.offset = 0
                if len(out) < nbytes:
                    out.extend(b"\x00" * (nbytes - len(out)))
                outdata[:] = bytes(out[:nbytes])

        self.stream = self.sd.RawOutputStream(
            samplerate=self.rate,
            channels=self.channels,
            dtype="int16",
            blocksize=0,
            latency="low",
            callback=callback,
        )
        self.stream.start()
        self.active = True

    def set_slice(self, wav: WavData, start_sec: float, end_sec: float, playback_rate: float = 1.0, reverse: bool = False):
        self.start(wav, playback_rate)
        start_sec = max(0.0, min(start_sec, wav.duration))
        end_sec = max(0.0, min(end_sec, wav.duration))
        if end_sec < start_sec:
            start_sec, end_sec = end_sec, start_sec
            reverse = not reverse
        frame_size = wav.channels * wav.sample_width
        start_frame = int(start_sec * wav.rate)
        end_frame = max(start_frame + 1, int(end_sec * wav.rate))
        a = max(0, start_frame * frame_size)
        b = min(len(wav.raw), end_frame * frame_size)
        data = wav.raw[a:b]
        if reverse and data:
            frames = [data[i:i+frame_size] for i in range(0, len(data), frame_size)]
            frames.reverse()
            data = b"".join(frames)
        # Very tiny slices make an inaudible click. Pad to at least ~20ms,
        # but the callback still consumes this padded slice once and then
        # outputs silence until the next mouse-motion update.
        min_bytes = max(frame_size, int(wav.rate * 0.020) * frame_size)
        if data and len(data) < min_bytes:
            reps = (min_bytes + len(data) - 1) // len(data)
            data = (data * reps)[:min_bytes]
        with self._lock:
            self.slice_bytes = data
            self.offset = 0

    def stop(self):
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
        self.stream = None
        self.active = False
        self.slice_bytes = b""
        self.offset = 0

class Player:
    def __init__(self):
        self.proc: Optional[subprocess.Popen] = None
        self.temp_path: Optional[Path] = None
        self.started_at = 0.0
        self.start_sec = 0.0
        self.segment_duration = 0.0
        self.playback_rate = 1.0
        self.update_cursor = True
        self._temp_counter = 0
        self.is_windows = platform.system().lower().startswith("win")
        if self.is_windows:
            import winsound  # type: ignore
            self.winsound = winsound
        else:
            self.winsound = None

    def play_from(self, wav: WavData, start_sec: float, playback_rate: float = 1.0, duration_sec: Optional[float] = None, update_cursor: bool = True):
        self.stop()
        self.playback_rate = max(0.25, min(2.0, playback_rate))
        self.update_cursor = update_cursor
        self._temp_counter = (self._temp_counter + 1) % 100000
        tmp = Path(tempfile.gettempdir()) / f"mfa_review_play_{os.getpid()}_{self._temp_counter}.wav"
        self.segment_duration = wav.write_segment(start_sec, tmp, self.playback_rate, duration_sec=duration_sec)
        self.temp_path = tmp
        self.start_sec = start_sec
        self.started_at = time.time()
        if self.is_windows:
            self.winsound.PlaySound(str(tmp), self.winsound.SND_FILENAME | self.winsound.SND_ASYNC)
        else:
            cmd = None
            for c in ("afplay", "paplay", "aplay"):
                if shutil.which(c):
                    cmd = c
                    break
            if not cmd:
                raise RuntimeError("No playback command found. Install afplay, paplay, or aplay, or run on Windows.")
            self.proc = subprocess.Popen([cmd, str(tmp)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def play_slice(self, wav: WavData, start_sec: float, end_sec: float, playback_rate: float = 1.0, reverse: bool = False, update_cursor: bool = False):
        self.stop()
        self.playback_rate = max(0.25, min(2.0, playback_rate))
        self.update_cursor = update_cursor
        self._temp_counter = (self._temp_counter + 1) % 100000
        tmp = Path(tempfile.gettempdir()) / f"mfa_review_scrub_{os.getpid()}_{self._temp_counter}.wav"
        self.segment_duration = wav.write_slice(start_sec, end_sec, tmp, self.playback_rate, reverse=reverse)
        self.temp_path = tmp
        self.start_sec = min(start_sec, end_sec)
        self.started_at = time.time()
        if self.is_windows:
            self.winsound.PlaySound(str(tmp), self.winsound.SND_FILENAME | self.winsound.SND_ASYNC)
        else:
            cmd = None
            for c in ("afplay", "paplay", "aplay"):
                if shutil.which(c):
                    cmd = c
                    break
            if not cmd:
                raise RuntimeError("No playback command found. Install afplay, paplay, or aplay, or run on Windows.")
            self.proc = subprocess.Popen([cmd, str(tmp)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def stop(self):
        if self.is_windows and getattr(self, "winsound", None):
            self.winsound.PlaySound(None, self.winsound.SND_PURGE)
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
        self.proc = None

    def current_time(self) -> Optional[float]:
        if not self.started_at or not self.update_cursor:
            return None
        elapsed_audio = (time.time() - self.started_at) * self.playback_rate
        t = self.start_sec + elapsed_audio
        if time.time() - self.started_at > self.segment_duration:
            self.started_at = 0.0
            return None
        return t

class App:
    def __init__(self, root: tk.Tk, dataset_root: Path):
        self.root = root
        self.dataset_root = dataset_root.resolve()
        self.root.title("MFA Alignment Review Tool")
        self.cases = discover_cases(self.dataset_root)
        if not self.cases:
            raise RuntimeError(f"No cases found under {self.dataset_root}")
        self.case_index = 0
        self.wav: Optional[WavData] = None
        self.rows: Dict[str, List[Dict[str, str]]] = {"words": [], "visemes": [], "speech": []}
        self.csv_fields: Dict[str, List[str]] = {}
        self.dirty = False
        self.cursor_sec = 0.0
        self.zoom_start = 0.0
        self.zoom_end = 1.0
        self.drag = None
        self.scrub_drag = False
        self.last_scrub_seek_wall = 0.0
        self.scrub_prev_sec = 0.0
        self.scrub_last_motion_sec = 0.0
        self.scrub_enabled_var = tk.BooleanVar(value=True)
        self.player = Player()
        self.scrubber = ContinuousScrubber()
        self.playback_rate_var = tk.StringVar(value="1.00")
        self.edit_start_var = tk.StringVar()
        self.edit_end_var = tk.StringVar()
        self.selected = None  # (layer, row_index)
        self.show_layers = {"words": tk.BooleanVar(value=True), "visemes": tk.BooleanVar(value=True), "speech": tk.BooleanVar(value=True)}
        self._build_ui()
        self.load_case(0)
        self._tick()

    def _build_ui(self):
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=8, pady=6)
        self.case_combo = ttk.Combobox(top, width=72, state="readonly")
        self.case_combo.pack(side="left", fill="x", expand=True)
        self.case_combo["values"] = [c.case_id for c in self.cases]
        self.case_combo.bind("<<ComboboxSelected>>", lambda e: self.load_case(self.case_combo.current()))
        ttk.Button(top, text="Prev", command=lambda: self.load_case(max(0, self.case_index-1))).pack(side="left", padx=2)
        ttk.Button(top, text="Next", command=lambda: self.load_case(min(len(self.cases)-1, self.case_index+1))).pack(side="left", padx=2)
        ttk.Button(top, text="Save", command=self.save).pack(side="left", padx=2)

        self.transcript_var = tk.StringVar()
        ttk.Label(self.root, textvariable=self.transcript_var, wraplength=1100).pack(fill="x", padx=8)

        controls = ttk.Frame(self.root)
        controls.pack(fill="x", padx=8, pady=4)
        ttk.Button(controls, text="◀ 0.5s", command=lambda: self.seek(self.cursor_sec-0.5)).pack(side="left")
        ttk.Button(controls, text="Play", command=self.play).pack(side="left", padx=2)
        ttk.Button(controls, text="Stop", command=self.stop).pack(side="left", padx=2)
        ttk.Label(controls, text="Rate").pack(side="left", padx=(8,2))
        rate_box = ttk.Combobox(controls, width=5, textvariable=self.playback_rate_var, values=("0.50", "0.75", "1.00", "1.25", "1.50", "2.00"))
        rate_box.pack(side="left")
        rate_box.bind("<<ComboboxSelected>>", lambda e: self.restart_playback_if_active())
        rate_box.bind("<Return>", lambda e: self.restart_playback_if_active())
        ttk.Button(controls, text="0.5s ▶", command=lambda: self.seek(self.cursor_sec+0.5)).pack(side="left")
        ttk.Checkbutton(controls, text="Premiere scrub", variable=self.scrub_enabled_var).pack(side="left", padx=(10,8))
        ttk.Button(controls, text="Zoom In", command=lambda: self.zoom(0.5)).pack(side="left", padx=(16,2))
        ttk.Button(controls, text="Zoom Out", command=lambda: self.zoom(2.0)).pack(side="left")
        ttk.Button(controls, text="Fit", command=self.fit).pack(side="left", padx=2)
        for layer in ("words", "visemes", "speech"):
            ttk.Checkbutton(controls, text=layer, variable=self.show_layers[layer], command=self.redraw).pack(side="left", padx=6)
        self.time_var = tk.StringVar()
        ttk.Label(controls, textvariable=self.time_var).pack(side="right")

        self.canvas = tk.Canvas(self.root, height=420, background="white")
        self.canvas.pack(fill="both", expand=True, padx=8, pady=4)
        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Double-Button-1>", self.on_double_click)

        bottom = ttk.Frame(self.root)
        bottom.pack(fill="both", expand=False, padx=8, pady=6)
        self.tree = ttk.Treeview(bottom, columns=("layer","start","end","label","word","extra"), show="headings", height=10)
        for col, width in (("layer",80),("start",90),("end",90),("label",140),("word",160),("extra",260)):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=width, stretch=(col=="extra"))
        self.tree.pack(side="left", fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)
        sb = ttk.Scrollbar(bottom, command=self.tree.yview)
        sb.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=sb.set)

        edit = ttk.Frame(self.root)
        edit.pack(fill="x", padx=8, pady=(0,6))
        ttk.Label(edit, text="Selected marker start").pack(side="left")
        start_entry = ttk.Entry(edit, width=10, textvariable=self.edit_start_var)
        start_entry.pack(side="left", padx=(4,10))
        ttk.Label(edit, text="end").pack(side="left")
        end_entry = ttk.Entry(edit, width=10, textvariable=self.edit_end_var)
        end_entry.pack(side="left", padx=(4,10))
        ttk.Button(edit, text="Apply Marker Edit", command=self.apply_marker_edit).pack(side="left", padx=2)
        ttk.Button(edit, text="Start -1ms", command=lambda: self.nudge_selected("start", -0.001)).pack(side="left", padx=(12,2))
        ttk.Button(edit, text="Start +1ms", command=lambda: self.nudge_selected("start", 0.001)).pack(side="left", padx=2)
        ttk.Button(edit, text="End -1ms", command=lambda: self.nudge_selected("end", -0.001)).pack(side="left", padx=(12,2))
        ttk.Button(edit, text="End +1ms", command=lambda: self.nudge_selected("end", 0.001)).pack(side="left", padx=2)

        self.root.bind("<space>", lambda e: self.play())
        self.root.bind("<Left>", lambda e: self.seek(self.cursor_sec-0.1))
        self.root.bind("<Right>", lambda e: self.seek(self.cursor_sec+0.1))
        self.root.bind("<Control-Left>", lambda e: self.nudge_selected("start", -0.001))
        self.root.bind("<Control-Right>", lambda e: self.nudge_selected("start", 0.001))
        self.root.bind("<Shift-Left>", lambda e: self.nudge_selected("end", -0.001))
        self.root.bind("<Shift-Right>", lambda e: self.nudge_selected("end", 0.001))
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def load_case(self, idx: int):
        if self.dirty and not messagebox.askyesno("Unsaved edits", "Discard unsaved edits and switch cases?"):
            return
        self.stop()
        self.case_index = idx
        c = self.cases[idx]
        self.case_combo.current(idx)
        self.wav = WavData(c.wav_path)
        self.zoom_start = 0.0
        self.zoom_end = self.wav.duration
        self.cursor_sec = 0.0
        self.rows = {"words": [], "visemes": [], "speech": []}
        self.csv_fields = {}
        for layer, filename in (("words","words.csv"),("visemes","visemes.csv"),("speech","speech.csv")):
            # New gold exports may include canonical phones.csv.  Load it into the
            # existing visemes layer so older datasets and UI controls remain compatible.
            if layer == "visemes" and (c.gold_dir / "phones.csv").exists():
                filename = "phones.csv"
            p = c.gold_dir / filename
            if p.exists():
                with p.open(newline="", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    self.csv_fields[layer] = list(reader.fieldnames or [])
                    self.rows[layer] = [dict(r) for r in reader]
        text = c.transcript_path.read_text(encoding="utf-8-sig").strip() if c.transcript_path and c.transcript_path.exists() else ""
        self.transcript_var.set(text)
        self.dirty = False
        self.refresh_table()
        self.update_edit_fields()
        self.redraw()

    def refresh_table(self):
        self.tree.delete(*self.tree.get_children())
        for layer in ("speech", "words", "visemes"):
            for i, r in enumerate(self.rows[layer]):
                label = marker_label(r)
                word = r.get("word", "")
                extra = ", ".join(f"{k}={v}" for k,v in r.items() if k not in ("start","end","pose","word"))
                self.tree.insert("", "end", iid=f"{layer}:{i}", values=(layer, r.get("start",""), r.get("end",""), label, word, extra))

    def t_to_x(self, t: float) -> float:
        w = max(1, self.canvas.winfo_width())
        return (t - self.zoom_start) / max(1e-9, self.zoom_end - self.zoom_start) * w

    def x_to_t(self, x: float) -> float:
        return self.zoom_start + x / max(1, self.canvas.winfo_width()) * (self.zoom_end - self.zoom_start)

    def redraw(self):
        self.canvas.delete("all")
        if not self.wav:
            return
        w, h = max(1,self.canvas.winfo_width()), max(1,self.canvas.winfo_height())
        mid = h * 0.43
        amp_h = h * 0.36
        self.canvas.create_line(0, mid, w, mid, fill="#ddd")
        s0 = max(0, int(self.zoom_start * self.wav.rate))
        s1 = min(len(self.wav.samples), int(self.zoom_end * self.wav.rate))
        visible = self.wav.samples[s0:s1]
        if visible:
            step = max(1, len(visible) // w)
            pts = []
            for px in range(w):
                a = px * step
                b = min(len(visible), a + step)
                if a >= len(visible): break
                chunk = visible[a:b]
                mn, mx = min(chunk), max(chunk)
                pts.append((px, mid - mx * amp_h, px, mid - mn * amp_h))
            for x, y1, _, y2 in pts:
                self.canvas.create_line(x, y1, x, y2, fill="#444")
        # second bands
        band_y = {"speech": h*0.78, "words": h*0.86, "visemes": h*0.94}
        for layer in ("speech","words","visemes"):
            if not self.show_layers[layer].get(): continue
            color = MARKER_COLORS[layer]
            y = band_y[layer]
            self.canvas.create_text(6, y-12, text=layer, anchor="w", fill=color)
            for i, r in enumerate(self.rows[layer]):
                try:
                    a, b = float(r.get("start", "0")), float(r.get("end", "0"))
                except ValueError:
                    continue
                if b < self.zoom_start or a > self.zoom_end: continue
                x1, x2 = self.t_to_x(a), self.t_to_x(b)
                is_sel = self.selected == (layer, i)
                self.canvas.create_rectangle(x1, y-9, x2, y+9, outline=color, fill="#f8fafc" if not is_sel else "#fde68a", tags=(f"mark:{layer}:{i}",))
                self.canvas.create_line(x1, 0, x1, h, fill=color, stipple="gray50")
                self.canvas.create_line(x2, 0, x2, h, fill=color, stipple="gray75")
                label = marker_label(r)
                if x2 - x1 > 18:
                    self.canvas.create_text((x1+x2)/2, y, text=label[:18], fill=color, font=("TkDefaultFont", 8))
        cx = self.t_to_x(self.cursor_sec)
        self.canvas.create_line(cx, 0, cx, h, fill="black", width=2)
        # ticks
        tick = nice_tick((self.zoom_end - self.zoom_start) / 8)
        t = math.ceil(self.zoom_start / tick) * tick
        while t <= self.zoom_end:
            x = self.t_to_x(t)
            self.canvas.create_line(x, 0, x, 8, fill="#888")
            self.canvas.create_text(x+2, 10, text=f"{t:.2f}s", anchor="nw", fill="#666")
            t += tick
        self.time_var.set(f"{self.cursor_sec:.3f}s / {self.wav.duration:.3f}s")

    def on_click(self, event):
        t = self.x_to_t(event.x)
        nearest = self.find_nearest_boundary(event.x, event.y)
        if nearest:
            layer, idx, side = nearest
            self.selected = (layer, idx)
            self.drag = (layer, idx, side)
            self.select_tree(layer, idx)
            self.update_edit_fields()
        else:
            self.scrub_drag = True
            self.seek(t, stop=True)
            self.begin_premiere_scrub(t)
        self.redraw()

    def on_double_click(self, event):
        self.seek(self.x_to_t(event.x))
        self.play()

    def find_nearest_boundary(self, x, y):
        best = None
        best_dist = 10
        for layer in ("speech","words","visemes"):
            if not self.show_layers[layer].get(): continue
            for i, r in enumerate(self.rows[layer]):
                try: a,b = float(r.get("start","0")), float(r.get("end","0"))
                except ValueError: continue
                for side, tt in (("start",a),("end",b)):
                    xx = self.t_to_x(tt)
                    d = abs(xx - x)
                    if d < best_dist:
                        best = (layer, i, side); best_dist = d
        return best

    def on_drag(self, event):
        if not self.wav:
            return
        if self.drag:
            layer, idx, side = self.drag
            r = self.rows[layer][idx]
            t = max(0.0, min(self.wav.duration, self.x_to_t(event.x)))
            try:
                other = float(r["end" if side == "start" else "start"])
            except Exception:
                other = t
            if side == "start":
                t = min(t, other - 0.001)
            else:
                t = max(t, other + 0.001)
            r[side] = f"{t:.6f}"
            self.cursor_sec = t
            self.dirty = True
            self.refresh_table()
            self.select_tree(layer, idx)
            self.update_edit_fields()
            self.redraw()
            return
        if self.scrub_drag:
            new_t = max(0.0, min(self.wav.duration, self.x_to_t(event.x)))
            self.cursor_sec = new_t
            self.ensure_cursor_visible()
            self.scrub_to(new_t)
            self.redraw()

    def on_release(self, event):
        was_scrubbing = self.scrub_drag
        self.drag = None
        self.scrub_drag = False
        if was_scrubbing:
            self.scrubber.stop()
            self.stop()

    def on_tree_select(self, event):
        sel = self.tree.selection()
        if not sel: return
        layer, idxs = sel[0].split(":")
        idx = int(idxs)
        self.selected = (layer, idx)
        self.update_edit_fields()
        r = self.rows[layer][idx]
        try: self.seek(float(r.get("start","0")), stop=False)
        except ValueError: pass
        self.redraw()

    def update_edit_fields(self):
        if not self.selected:
            self.edit_start_var.set("")
            self.edit_end_var.set("")
            return
        layer, idx = self.selected
        if layer not in self.rows or idx < 0 or idx >= len(self.rows[layer]):
            self.edit_start_var.set("")
            self.edit_end_var.set("")
            return
        r = self.rows[layer][idx]
        self.edit_start_var.set(r.get("start", ""))
        self.edit_end_var.set(r.get("end", ""))

    def apply_marker_edit(self):
        if not self.selected or not self.wav:
            messagebox.showinfo("No marker selected", "Select a marker in the table or click one on the waveform first.")
            return
        layer, idx = self.selected
        r = self.rows[layer][idx]
        try:
            start = float(self.edit_start_var.get())
            end = float(self.edit_end_var.get())
        except ValueError:
            messagebox.showerror("Invalid marker edit", "Start and end must be numeric seconds, e.g. 1.234")
            return
        start = max(0.0, min(self.wav.duration, start))
        end = max(0.0, min(self.wav.duration, end))
        if end <= start:
            messagebox.showerror("Invalid marker edit", "End must be after start.")
            return
        r["start"] = f"{start:.6f}"
        r["end"] = f"{end:.6f}"
        self.cursor_sec = start
        self.dirty = True
        self.refresh_table()
        self.select_tree(layer, idx)
        self.update_edit_fields()
        self.redraw()

    def nudge_selected(self, side: str, delta: float):
        if not self.selected or not self.wav:
            return
        layer, idx = self.selected
        r = self.rows[layer][idx]
        try:
            start = float(r.get("start", "0"))
            end = float(r.get("end", "0"))
        except ValueError:
            return
        if side == "start":
            start = max(0.0, min(end - 0.001, start + delta))
            self.cursor_sec = start
        else:
            end = min(self.wav.duration, max(start + 0.001, end + delta))
            self.cursor_sec = end
        r["start"] = f"{start:.6f}"
        r["end"] = f"{end:.6f}"
        self.dirty = True
        self.refresh_table()
        self.select_tree(layer, idx)
        self.update_edit_fields()
        self.redraw()

    def select_tree(self, layer, idx):
        iid = f"{layer}:{idx}"
        if self.tree.exists(iid):
            self.tree.selection_set(iid)
            self.tree.see(iid)

    def seek(self, t: float, stop: bool=True):
        if not self.wav: return
        self.cursor_sec = max(0.0, min(self.wav.duration, t))
        if stop: self.stop()
        self.ensure_cursor_visible()
        self.redraw()

    def ensure_cursor_visible(self):
        if self.cursor_sec < self.zoom_start or self.cursor_sec > self.zoom_end:
            span = self.zoom_end - self.zoom_start
            self.zoom_start = max(0.0, self.cursor_sec - span*0.25)
            self.zoom_end = min(self.wav.duration if self.wav else self.cursor_sec + span, self.zoom_start + span)

    def get_playback_rate(self) -> float:
        try:
            return max(0.25, min(2.0, float(self.playback_rate_var.get())))
        except ValueError:
            self.playback_rate_var.set("1.00")
            return 1.0

    def begin_premiere_scrub(self, t: float):
        if not self.wav or not self.scrub_enabled_var.get():
            return
        now = time.time()
        self.scrub_prev_sec = t
        self.scrub_last_motion_sec = t
        self.last_scrub_seek_wall = now
        self.play_scrub_slice(t - 0.030, t + 0.030, reverse=False)

    def scrub_to(self, new_t: float):
        if not self.wav or not self.scrub_enabled_var.get():
            return
        now = time.time()
        # Update often enough that very slow drags still produce audio.  The
        # sounddevice path can tolerate frequent replacement slices; the temp-WAV
        # fallback remains best-effort.
        if now - self.last_scrub_seek_wall < 0.004:
            return

        old_t = self.scrub_last_motion_sec
        dt = new_t - old_t

        # Only true stationary hold should be silent.  Do not require a large
        # time jump: when zoomed in, a careful mouse drag may move less than a
        # millisecond per event, but that is exactly the desired scrub gesture.
        if abs(dt) < 0.00025:
            return

        self.last_scrub_seek_wall = now
        self.scrub_last_motion_sec = new_t

        # Premiere-style scrub: audible material follows the direction of
        # playhead travel. For slow movement the swept interval may be too tiny
        # to hear, so extend the slice ahead of the motion direction rather than
        # waiting for a larger cursor jump. Holding still still produces no new
        # slices, so it remains silent.
        min_span = 0.045
        if dt > 0:
            a = old_t
            b = max(new_t, old_t + min_span)
            reverse = False
        else:
            b = old_t
            a = min(new_t, old_t - min_span)
            reverse = True

        a = max(0.0, min(self.wav.duration, a))
        b = max(0.0, min(self.wav.duration, b))
        if b <= a:
            return
        self.play_scrub_slice(a, b, reverse=reverse)

    def play_scrub_slice(self, start_sec: float, end_sec: float, reverse: bool = False):
        if not self.wav:
            return
        try:
            self.scrubber.set_slice(self.wav, start_sec, end_sec, self.get_playback_rate(), reverse=reverse)
            return
        except Exception as e:
            # Fallback to the old temp-WAV approach if sounddevice is not installed
            # or the audio device cannot open. It is less smooth, but keeps the
            # rest of the tool usable.
            if not getattr(self, "_warned_scrub_fallback", False):
                self._warned_scrub_fallback = True
                print(f"Continuous scrub unavailable; falling back to temp-WAV scrub: {e}", file=sys.stderr)
        try:
            # Avoid flooding winsound/afplay in fallback mode.
            self.player.play_slice(self.wav, start_sec, end_sec, self.get_playback_rate(), reverse=reverse, update_cursor=False)
        except Exception:
            pass

    def restart_playback_if_active(self):
        if self.player.current_time() is not None:
            self.play()

    def play(self):
        if not self.wav: return
        try:
            self.player.play_from(self.wav, self.cursor_sec, self.get_playback_rate())
        except Exception as e:
            messagebox.showerror("Playback failed", str(e))

    def play_slice(self, wav: WavData, start_sec: float, end_sec: float, playback_rate: float = 1.0, reverse: bool = False, update_cursor: bool = False):
        self.stop()
        self.playback_rate = max(0.25, min(2.0, playback_rate))
        self.update_cursor = update_cursor
        self._temp_counter = (self._temp_counter + 1) % 100000
        tmp = Path(tempfile.gettempdir()) / f"mfa_review_scrub_{os.getpid()}_{self._temp_counter}.wav"
        self.segment_duration = wav.write_slice(start_sec, end_sec, tmp, self.playback_rate, reverse=reverse)
        self.temp_path = tmp
        self.start_sec = min(start_sec, end_sec)
        self.started_at = time.time()
        if self.is_windows:
            self.winsound.PlaySound(str(tmp), self.winsound.SND_FILENAME | self.winsound.SND_ASYNC)
        else:
            cmd = None
            for c in ("afplay", "paplay", "aplay"):
                if shutil.which(c):
                    cmd = c
                    break
            if not cmd:
                raise RuntimeError("No playback command found. Install afplay, paplay, or aplay, or run on Windows.")
            self.proc = subprocess.Popen([cmd, str(tmp)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def stop(self):
        if hasattr(self, "scrubber"):
            self.scrubber.stop()
        self.player.stop()

    def zoom(self, factor: float):
        if not self.wav: return
        center = self.cursor_sec
        span = max(0.05, min(self.wav.duration, (self.zoom_end - self.zoom_start) * factor))
        self.zoom_start = max(0.0, center - span/2)
        self.zoom_end = min(self.wav.duration, self.zoom_start + span)
        self.zoom_start = max(0.0, self.zoom_end - span)
        self.redraw()

    def fit(self):
        if self.wav:
            self.zoom_start = 0.0
            self.zoom_end = self.wav.duration
            self.redraw()

    def save(self):
        c = self.cases[self.case_index]
        stamp = time.strftime("%Y%m%d_%H%M%S")
        for layer, filename in (("words","words.csv"),("visemes","visemes.csv"),("speech","speech.csv")):
            if layer == "visemes" and (c.gold_dir / "phones.csv").exists():
                filename = "phones.csv"
            path = c.gold_dir / filename
            if not path.exists():
                continue
            backup = path.with_suffix(path.suffix + f".bak_{stamp}")
            if not backup.exists():
                shutil.copy2(path, backup)
            fields = self.csv_fields.get(layer) or list(self.rows[layer][0].keys() if self.rows[layer] else [])
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for r in self.rows[layer]:
                    writer.writerow({k: r.get(k, "") for k in fields})
        self.dirty = False
        messagebox.showinfo("Saved", "CSV files saved. Original files were backed up next to each CSV.")

    def _tick(self):
        t = self.player.current_time()
        if t is not None:
            self.cursor_sec = t
            self.ensure_cursor_visible()
            self.redraw()
        self.root.after(33, self._tick)

    def close(self):
        if self.dirty and not messagebox.askyesno("Unsaved edits", "Quit without saving?"):
            return
        self.stop()
        self.root.destroy()

def nice_tick(x: float) -> float:
    if x <= 0: return 0.1
    p = 10 ** math.floor(math.log10(x))
    for m in (1, 2, 5, 10):
        if m * p >= x:
            return m * p
    return p

def discover_cases(root: Path) -> List[Case]:
    wav_dir = root / "wav"
    gold_dir = root / "gold"
    trans_dir = root / "transcripts"
    cases: List[Case] = []
    if wav_dir.exists() and gold_dir.exists():
        for wav_path in sorted(wav_dir.glob("*.wav")):
            cid = wav_path.stem
            gd = gold_dir / cid
            if not gd.exists():
                continue
            transcript = trans_dir / f"{cid}.txt"
            cases.append(Case(cid, wav_path, transcript if transcript.exists() else None, gd))
    if cases:
        return cases
    # fallback: search recursively for wavs with nearby gold/<stem>
    for wav_path in sorted(root.rglob("*.wav")):
        cid = wav_path.stem
        matches = list(root.rglob(f"gold/{cid}"))
        if matches:
            transcript = next(iter(root.rglob(f"{cid}.txt")), None)
            cases.append(Case(cid, wav_path, transcript, matches[0]))
    return cases

def main():
    dataset = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    root = tk.Tk()
    root.geometry("1280x820")
    try:
        App(root, dataset)
    except Exception as e:
        messagebox.showerror("Startup failed", str(e))
        raise
    root.mainloop()

if __name__ == "__main__":
    main()
