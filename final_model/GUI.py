#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
MindScribe — Live (Cyton/Synthetic) with EMG-driven, doctor-style conversation
Vocabulary is restricted to: {Pain, Yes, No, HMMM}

Key features:
- GUI sends a 'gui_ready' signal; main waits before starting streams.
- Gate detection disabled until the ring has >= PRE_SAMP samples.
- Board switch UI defaults to SYNTHETIC to match CONFIG.BOARD_MODE.
- Live camera feed (top-right).
- Blink-to-confirm (MediaPipe FaceMesh; EAR-based) for underconfident, non-urgent words.
- Text-to-Speech (TTS) with Kokoro-82M first (normalized), then pyttsx3 fallback.

Dependencies (install at least once):
    pip install kokoro sounddevice pyttsx3 mediapipe opencv-python pillow
"""

import os, sys, time, signal, queue, copy
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")




try:
    import torch
    import torch.nn.functional as F
    try:
        torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "1")))
    except Exception:
        pass
    try:
        import torch.backends.mkldnn as _mkldnn
        _mkldnn.enabled = False
    except Exception:
        pass
except Exception as e:
    raise SystemExit("PyTorch is required. pip install torch") from e

try:
    from braindecode.models import ATCNet
except Exception as e:
    raise SystemExit("braindecode is required. pip install braindecode") from e

from scipy.signal import butter, filtfilt, iirnotch, resample
from scipy.ndimage import uniform_filter1d




try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    _HAS_BRAINFLOW = True
except Exception:
    BoardShim = BrainFlowInputParams = BoardIds = None
    _HAS_BRAINFLOW = False

from multiprocessing import Process, Queue





class CONFIG:

    CKPT_PATH = r"C:\Users\krivi\OneDrive\Desktop\Mindscribe_fromscratch\main_September_26th\final_model\Yes_No_Krivan_3channels\best_model.pt"
    SERIAL    = "COM6"
    BOARD_MODE = "SYNTHETIC"


    USED_CHANNELS  = [1,2,3,4,5,6,7,8]
    MODEL_CHANNELS = list(USED_CHANNELS)
    GATE_CHANNELS  = [1,2,3,4]


    PRE_SAMP  = 800
    POST_SAMP = 1200


    COOLDOWN_SAMP   = 2500
    REFRACTORY_SAMP = None


    HP_CUTOFF     = 15.0
    NOTCH_HZ      = 0.0
    USE_UV_GATE   = True
    ZSCORE_THRESH = 4.5


    MIN_ACTIVE_CHANNELS = 2
    GATE_CONSEC_SAMP    = 1


    SYNTHETIC_Z_THRESH  = 1.5
    SYNTHETIC_MIN_ACTIVE = 1
    SYNTHETIC_CONSEC     = 3


    NO_GATE_FALLBACK_SEC = 15.0


    TOPK = 5
    FS_OVERRIDE = None


    RING_FACTOR = 3


    GUI_TITLE   = "MindScribe — Live"
    GUI_POLL_MS = 100
    LIVE_PLOT_SEC = 5.0


    VOICE_CAPTURE_SEC = 2.0
    VOICE_STEP_SEC    = 0.20


    ALLOWED_RESPONSES = ["pain","yes","no","hmmm"]


    ENABLE_CONVERSATION = True
    CONVERSATION_SPEC = {
        "start": "q_pain_now",
        "questions": {
            "q_pain_now":    {"type":"choice","text":"Pain now?","options":["pain","yes","no","hmmm"],
                               "followups":{"pain":"q_worse_today","yes":"q_worse_today","no":"q_breathing","hmmm":"q_breathing","_any":"q_breathing"}},
            "q_worse_today": {"type":"choice","text":"Worse than yesterday?","options":["pain","yes","no","hmmm"],
                               "followups":{"pain":"q_breathing","yes":"q_breathing","no":"q_breathing","hmmm":"q_breathing","_any":"q_breathing"}},
            "q_breathing":   {"type":"choice","text":"Breathing hard?","options":["pain","yes","no","hmmm"],
                               "followups":{"pain":"q_alert","yes":"q_alert","no":"q_dizzy","hmmm":"q_dizzy","_any":"q_dizzy"}},
            "q_dizzy":       {"type":"choice","text":"Dizzy?","options":["pain","yes","no","hmmm"],
                               "followups":{"pain":"q_alert","yes":"q_alert","no":"q_fever","hmmm":"q_fever","_any":"q_fever"}},
            "q_fever":       {"type":"choice","text":"Fever today?","options":["pain","yes","no","hmmm"],
                               "followups":{"pain":"q_pain_relief","yes":"q_pain_relief","no":"q_pain_relief","hmmm":"q_pain_relief","_any":"q_pain_relief"}},
            "q_pain_relief": {"type":"choice","text":"Need pain relief?","options":["pain","yes","no","hmmm"],
                               "followups":{"pain":"q_rest","yes":"q_rest","no":"q_rest","hmmm":"q_rest","_any":"q_rest"}},
            "q_rest":        {"type":"choice","text":"Rest/fluids now?","options":["pain","yes","no","hmmm"],
                               "followups":{"pain":"q_alert","yes":"q_alert","no":"q_alert","hmmm":"q_alert","_any":"q_alert"}},
            "q_alert":       {"type":"choice","text":"Alert caregiver now?","options":["pain","yes","no","hmmm"],
                               "followups":{"pain":None,"yes":None,"no":"q_followup","hmmm":"q_followup","_any":"q_followup"}},
            "q_followup":    {"type":"choice","text":"Check again later?","options":["pain","yes","no","hmmm"],
                               "followups":{"pain":None,"yes":None,"no":None,"hmmm":None,"_any":None}}
        }
    }


    ENABLE_TTS = True
    TTS_PREFERRED = ("kokoro", "pyttsx3")


    TTS_RATE_WPM = 185
    TTS_VOLUME   = 0.9


    KOKORO_LANG   = "a"
    KOKORO_VOICE  = "af_heart"
    KOKORO_SPEED  = 1.0

    KOKORO_TARGET_RMS = 0.08
    KOKORO_CLIP_PEAK  = 0.98





class LiveBoard:
    def __init__(self, serial_port: str, board_id: str = "SYNTHETIC"):
        name = str(board_id).upper()
        self.mode = name
        if _HAS_BRAINFLOW:
            BoardShim.enable_dev_board_logger()
            params = BrainFlowInputParams()
            params.serial_port = serial_port or ""
            if name == "CYTON_DAISY": bid = BoardIds.CYTON_DAISY_BOARD.value
            elif name == "CYTON":     bid = BoardIds.CYTON_BOARD.value
            else:                     bid = BoardIds.SYNTHETIC_BOARD.value

            self.board = BoardShim(bid, params)
            self.board.prepare_session()
            self.board.start_stream()
            time.sleep(0.3)

            self.fs = float(BoardShim.get_sampling_rate(bid))
            chs = BoardShim.get_eeg_channels(bid)
            if not chs:
                try: chs = BoardShim.get_exg_channels(bid)
                except Exception: chs = []
            self.ch_idx = chs
            self.n_ch = len(chs)
            if self.n_ch == 0:
                raise RuntimeError(f"No EEG/EXG channels for board {name} (id={bid}).")
            self._chunk = max(1, int(self.fs * 0.12))
            print(f"[Live] Connected: board={name}, fs={self.fs:.1f} Hz, channels={self.ch_idx}")
        else:

            self.fs = 250.0
            self.n_ch = 8
            self.ch_idx = list(range(self.n_ch))
            self._chunk = max(1, int(self.fs * 0.12))
            self._t = 0
            print(f"[Live] BrainFlow not found → using built-in synthetic source @ {self.fs:.1f} Hz")

    def read_chunk(self) -> np.ndarray:
        if _HAS_BRAINFLOW and hasattr(self, "board"):
            data = self.board.get_current_board_data(self._chunk)
            if data.size == 0:
                return np.zeros((self.n_ch, 0), dtype=np.float32)
            return np.asarray(data[self.ch_idx, :], dtype=np.float32)

        N = self._chunk
        sig = 10.0 * np.random.randn(self.n_ch, N).astype(np.float32)
        self._t += N
        return sig

    def close(self):
        if _HAS_BRAINFLOW and hasattr(self, "board"):
            try: self.board.stop_stream()
            except Exception: pass
            try: self.board.release_session()
            except Exception: pass

LiveCyton = LiveBoard





def gui_process(to_gui: Queue, from_gui: Queue, title: str, poll_ms: int):
    import tkinter as tk
    from tkinter import ttk
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.pyplot as plt
    import numpy as np
    import threading
    import time as _time


    _HAS_CV2 = True
    _HAS_MEDIAPIPE = True
    _HAS_PIL = True

    try:
        import cv2
    except Exception as _e:
        print(f"[GUI] OpenCV not available: {_e}")
        _HAS_CV2 = False
    try:
        import mediapipe as mp
    except Exception as _e:
        print(f"[GUI] MediaPipe not available (blink off): {_e}")
        _HAS_MEDIAPIPE = False
    try:
        from PIL import Image, ImageTk
    except Exception as _e:
        print(f"[GUI] Pillow not available (camera preview off): {_e}")
        _HAS_PIL = False


    _HAS_KOKORO = True
    _HAS_SD     = True
    _HAS_PYTTSX3= True
    try:
        from kokoro import KPipeline
    except Exception as _e:
        print(f"[TTS] Kokoro not available: {_e}")
        _HAS_KOKORO = False
    try:
        import sounddevice as sd
    except Exception as _e:
        print(f"[TTS] sounddevice not available: {_e}")
        _HAS_SD = False
    try:
        import pyttsx3
    except Exception as _e:
        print(f"[TTS] pyttsx3 not available: {_e}")
        _HAS_PYTTSX3 = False

    root = tk.Tk()
    root.title(title)
    root.geometry("1500x900")


    main = ttk.Frame(root, padding=6); main.pack(fill=tk.BOTH, expand=True)
    sidebar = ttk.Frame(main, width=300); sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0,6)); sidebar.pack_propagate(False)
    center  = ttk.Frame(main); center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,6))
    right_col = ttk.Frame(main); right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)


    top = ttk.Frame(center, padding=6); top.pack(side=tk.TOP, fill=tk.X)
    status_var = tk.StringVar(value="Waiting for first inference…")
    info_var   = tk.StringVar(value="—")
    ttk.Label(top, textvariable=status_var, font=("Arial", 16)).pack(anchor="w")
    ttk.Label(top, textvariable=info_var,   font=("Arial", 11)).pack(anchor="w")

    board_var = tk.StringVar(value="SYNTHETIC")
    switch_row = ttk.Frame(top); switch_row.pack(anchor="w", pady=(6,0))
    ttk.Label(switch_row, text="Board:").pack(side=tk.LEFT)
    ttk.Radiobutton(switch_row, text="CYTON", value="CYTON", variable=board_var,
                    command=lambda: from_gui.put({"type": "switch_board", "board": board_var.get()})).pack(side=tk.LEFT, padx=(8,0))
    ttk.Radiobutton(switch_row, text="SYNTHETIC", value="SYNTHETIC", variable=board_var,
                    command=lambda: from_gui.put({"type": "switch_board", "board": board_var.get()})).pack(side=tk.LEFT, padx=(8,0))


    fig_prob = plt.Figure(figsize=(8.5, 3.5)); ax_prob = fig_prob.add_subplot(111)
    ax_prob.set_title("Last inference — class probability distribution")
    ax_prob.set_ylabel("Probability"); ax_prob.grid(True, axis="y", alpha=0.25)
    canvas_prob = FigureCanvasTkAgg(fig_prob, master=center); canvas_prob.draw()
    canvas_prob.get_tk_widget().pack(fill=tk.BOTH, expand=False)


    fig_sig = plt.Figure(figsize=(8.5, 3.8)); ax_sig = fig_sig.add_subplot(111)
    ax_sig.set_title("Live signal (gate channels) — stacked, z-norm per channel")
    ax_sig.set_ylabel("Amplitude (a.u.)"); ax_sig.grid(True, axis="y", alpha=0.25); ax_sig.margins(y=0.10)
    canvas_sig = FigureCanvasTkAgg(fig_sig, master=center); canvas_sig.draw()
    canvas_sig.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(6,0))


    cam_frame = ttk.LabelFrame(right_col, text="Camera (live)")
    cam_frame.pack(side=tk.TOP, fill=tk.X, expand=False)
    cam_label = tk.Label(cam_frame, text="Camera feed will appear here.")
    cam_label.pack(side=tk.TOP, padx=6, pady=6)
    cam_label._photo = None
    CAM_PREVIEW_WIDTH = 420

    fig_last = plt.Figure(figsize=(6.0, 7.0)); ax_last = fig_last.add_subplot(111)
    ax_last.set_title("Last inference window (model channels)")
    ax_last.set_xlabel("Time (s)"); ax_last.set_ylabel("Amplitude (a.u., stacked)")
    ax_last.grid(True, axis="y", alpha=0.25); ax_last.margins(y=0.10)
    canvas_last = FigureCanvasTkAgg(fig_last, master=right_col); canvas_last.draw()
    canvas_last.get_tk_widget().pack(fill=tk.BOTH, expand=True)


    log = tk.Text(center, height=6); log.pack(fill=tk.BOTH, expand=False, pady=(6,0))
    def logln(s: str): log.insert(tk.END, s + "\n"); log.see(tk.END)
    logln("GUI ready. Waiting for updates…")


    ttk.Label(sidebar, text="Settings", font=("Arial", 13, "bold")).pack(anchor="w", pady=(4,6))
    gate_mode_var      = tk.StringVar(value="z")
    uv_threshold_var   = tk.DoubleVar(value=50.0)
    z_threshold_var    = tk.DoubleVar(value=4.5)
    min_active_var     = tk.IntVar(value=2)
    consec_var         = tk.IntVar(value=1)
    cooldown_var       = tk.IntVar(value=2500)

    gate_frame = ttk.LabelFrame(sidebar, text="Gate Settings"); gate_frame.pack(fill=tk.X, padx=0, pady=(0,8))
    mode_row = ttk.Frame(gate_frame); mode_row.pack(fill=tk.X, padx=6, pady=(6,2))
    ttk.Label(mode_row, text="Mode").pack(side=tk.LEFT)
    ttk.Radiobutton(mode_row, text="µV", value="uv", variable=gate_mode_var).pack(side=tk.LEFT, padx=(10,0))
    ttk.Radiobutton(mode_row, text="z-score", value="z", variable=gate_mode_var).pack(side=tk.LEFT, padx=(6,0))
    thr_row1 = ttk.Frame(gate_frame); thr_row1.pack(fill=tk.X, padx=6, pady=(4,2))
    ttk.Label(thr_row1, text="µV thr").pack(side=tk.LEFT)
    tk.Spinbox(thr_row1, from_=10, to=200, increment=5, width=6, textvariable=uv_threshold_var).pack(side=tk.LEFT, padx=(6,0))
    thr_row2 = ttk.Frame(gate_frame); thr_row2.pack(fill=tk.X, padx=6, pady=(2,2))
    ttk.Label(thr_row2, text="|z| thr").pack(side=tk.LEFT)
    tk.Spinbox(thr_row2, from_=0.5, to=10.0, increment=0.1, width=6, textvariable=z_threshold_var).pack(side=tk.LEFT, padx=(16,0))
    mac_row = ttk.Frame(gate_frame); mac_row.pack(fill=tk.X, padx=6, pady=(4,2))
    ttk.Label(mac_row, text="Min active").pack(side=tk.LEFT)
    tk.Spinbox(mac_row, from_=1, to=32, increment=1, width=4, textvariable=min_active_var).pack(side=tk.LEFT, padx=(6,12))
    ttk.Label(mac_row, text="Consec").pack(side=tk.LEFT)
    tk.Spinbox(mac_row, from_=1, to=20, increment=1, width=4, textvariable=consec_var).pack(side=tk.LEFT, padx=(6,0))
    cd_row = ttk.Frame(gate_frame); cd_row.pack(fill=tk.X, padx=6, pady=(4,6))
    ttk.Label(cd_row, text="Cooldown (samples)").pack(side=tk.LEFT)
    tk.Spinbox(cd_row, from_=0, to=20000, increment=50, width=7, textvariable=cooldown_var).pack(side=tk.LEFT, padx=(6,0))
    def apply_gate_settings():
        try:
            cfg = {"type":"config_update",
                   "use_uv_gate": (gate_mode_var.get() == "uv"),
                   "threshold_uV": float(uv_threshold_var.get()),
                   "zscore_thresh": float(z_threshold_var.get()),
                   "min_active_channels": int(min_active_var.get()),
                   "consec_samp": int(consec_var.get()),
                   "cooldown_samp": int(cooldown_var.get())}
            from_gui.put(cfg); logln("[UI] Gate settings sent.")
        except Exception as e:
            logln(f"[UI] Failed to send gate settings: {e}")
    ttk.Button(gate_frame, text="Apply Gate", command=apply_gate_settings).pack(fill=tk.X, padx=6, pady=(0,6))


    live_sec_var       = tk.DoubleVar(value=CONFIG.LIVE_PLOT_SEC)
    rebuild_buffer_var = tk.BooleanVar(value=False)
    paused_var         = tk.BooleanVar(value=False)
    live_frame = ttk.LabelFrame(sidebar, text="Live Plot"); live_frame.pack(fill=tk.X, padx=0, pady=(0,8))
    lp_row1 = ttk.Frame(live_frame); lp_row1.pack(fill=tk.X, padx=6, pady=(6,2))
    ttk.Label(lp_row1, text="Window (sec)").pack(side=tk.LEFT)
    tk.Spinbox(lp_row1, from_=0.5, to=30.0, increment=0.5, width=6, textvariable=live_sec_var).pack(side=tk.LEFT, padx=(6,12))
    ttk.Checkbutton(lp_row1, text="Rebuild buffer", variable=rebuild_buffer_var).pack(side=tk.LEFT)
    def set_live_window_sec(new_sec: float, rebuild: bool):
        nonlocal fs, live_sec, sig_ring, sig_wptr, sig_filled, sig_lines, n_ch_live, max_window_sec
        try: new_sec = float(new_sec)
        except Exception: return
        if rebuild and fs and n_ch_live:
            Tbuf = max(1, int(round(new_sec * fs)))
            sig_ring = np.zeros((n_ch_live, Tbuf), dtype=np.float32)
            sig_wptr = 0; sig_filled = 0; sig_lines = []
            ax_sig.clear(); ax_sig.set_title("Live signal (gate channels) — stacked, z-norm per channel")
            ax_sig.set_ylabel("Amplitude (a.u.)"); ax_sig.grid(True, axis="y", alpha=0.25); ax_sig.margins(y=0.10)
            for i in range(n_ch_live):
                ln, = ax_sig.plot([], [], linewidth=1.0, label=f"CH{i+1}"); sig_lines.append(ln)
            if n_ch_live <= 12: ax_sig.legend(loc="upper right", fontsize=8)
            canvas_sig.draw_idle()
        max_window_sec = max(0.5, new_sec); live_sec = new_sec
        logln(f"[UI] Live window set to {new_sec:.2f}s (rebuild={rebuild})")
    def apply_live_settings(): set_live_window_sec(live_sec_var.get(), rebuild=rebuild_buffer_var.get())
    def toggle_pause():
        nonlocal paused_live
        paused_live = bool(paused_var.get())
        logln(f"[UI] Live plot {'paused' if paused_live else 'resumed'}.")
    ttk.Button(live_frame, text="Apply Live Window", command=apply_live_settings).pack(fill=tk.X, padx=6, pady=(2,4))
    lp_row2 = ttk.Frame(live_frame); lp_row2.pack(fill=tk.X, padx=6, pady=(2,2))
    ttk.Checkbutton(lp_row2, text="Pause plot", variable=paused_var, command=toggle_pause).pack(side=tk.LEFT)
    def do_autoscale_sig(): ax_sig.relim(); ax_sig.autoscale_view(scalex=False, scaley=True); canvas_sig.draw_idle()
    def do_autoscale_last(): ax_last.relim(); ax_last.autoscale_view(scalex=False, scaley=True); canvas_last.draw_idle()
    ttk.Button(live_frame, text="Autoscale Y (live)", command=do_autoscale_sig).pack(fill=tk.X, padx=6, pady=(4,2))
    ttk.Button(live_frame, text="Autoscale Y (last)", command=do_autoscale_last).pack(fill=tk.X, padx=6, pady=(0,6))


    speech_frame = ttk.LabelFrame(sidebar, text="Speech (TTS)")
    speech_frame.pack(fill=tk.X, padx=0, pady=(0,8))
    tts_enabled_var = tk.BooleanVar(value=CONFIG.ENABLE_TTS)
    ttk.Checkbutton(speech_frame, text="Speak questions aloud", variable=tts_enabled_var).pack(anchor="w", padx=6, pady=(6,6))
    if not (_HAS_KOKORO and _HAS_SD) and not _HAS_PYTTSX3:
        logln("[TTS] No available backend. Install 'kokoro sounddevice' or 'pyttsx3'.")


    tts_queue = queue.Queue()
    tts_lock = threading.Lock()
    _HAS_TTS_BACKEND = (_HAS_KOKORO and _HAS_SD) or _HAS_PYTTSX3

    def _rms_normalize(x: np.ndarray, target_rms: float = CONFIG.KOKORO_TARGET_RMS, clip_peak: float = CONFIG.KOKORO_CLIP_PEAK) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        rms = float(np.sqrt(np.mean(np.square(x)) + 1e-12))
        if rms > 0.0:
            x = x * (target_rms / rms)
        pk = float(np.max(np.abs(x)) + 1e-9)
        if pk > clip_peak:
            x = x * (clip_peak / pk)
        return np.clip(x, -1.0, 1.0)

    def _tts_worker():
        kokoro = None
        engine = None
        sr = 24000


        if _HAS_KOKORO and _HAS_SD:
            try:
                kokoro = KPipeline(lang_code=CONFIG.KOKORO_LANG)
                print("[TTS] Using Kokoro-82M")
            except Exception as e:
                print(f"[TTS] Kokoro init failed: {e}")
                kokoro = None


        if kokoro is None and _HAS_PYTTSX3:
            try:
                engine = pyttsx3.init()
                try:
                    engine.setProperty('rate', int(CONFIG.TTS_RATE_WPM))
                    engine.setProperty('volume', float(CONFIG.TTS_VOLUME))
                except Exception:
                    pass
                print("[TTS] Using pyttsx3 fallback")
            except Exception as e:
                print(f"[TTS] pyttsx3 init failed: {e}")
                engine = None

        def _speak_kokoro(text: str) -> bool:
            if kokoro is None or not _HAS_SD: return False
            try:
                gen = kokoro(text, voice=CONFIG.KOKORO_VOICE, speed=float(CONFIG.KOKORO_SPEED), split_pattern=r"\n+")
                parts = []
                for _, _, audio in gen:
                    a = np.asarray(audio, dtype=np.float32).reshape(-1)
                    parts.append(a)
                if not parts: return False
                y = np.concatenate(parts)
                y = _rms_normalize(y)
                try: sd.stop()
                except Exception: pass
                sd.play(y, sr, blocking=True)
                return True
            except Exception as e:
                print(f"[TTS] Kokoro speak error: {e}")
                return False

        def _speak_py(text: str) -> bool:
            if engine is None: return False
            try: engine.stop()
            except Exception: pass
            try:
                engine.say(text)
                engine.runAndWait()
                return True
            except Exception as e:
                print(f"[TTS] pyttsx3 speak error: {e}")
                return False

        while True:
            item = tts_queue.get()
            if item is None:
                try:
                    if _HAS_SD: sd.stop()
                except Exception: pass
                try:
                    if engine is not None: engine.stop()
                except Exception: pass
                break

            cmd, payload = item
            if cmd == "speak":
                text = (payload or "").strip()
                if not text: continue

                if not _speak_kokoro(text):
                    _speak_py(text)
            elif cmd == "stop":
                try:
                    if _HAS_SD: sd.stop()
                except Exception: pass
                try:
                    if engine is not None: engine.stop()
                except Exception: pass

    if _HAS_TTS_BACKEND:
        threading.Thread(target=_tts_worker, daemon=True).start()

    def speak(text: str, flush: bool = True):
        """Queue text to speak; flushes pending items and preempts playback to avoid overlap (no 'loudness creep')."""
        if not (_HAS_TTS_BACKEND and tts_enabled_var.get() and text):
            return
        with tts_lock:
            if flush:
                try:

                    if _HAS_SD:
                        try: sd.stop()
                        except Exception: pass
                    while not tts_queue.empty():
                        try: tts_queue.get_nowait()
                        except queue.Empty: break
                except Exception:
                    pass
            try:
                tts_queue.put(("speak", text))
            except Exception as e:
                print(f"[TTS] queue error: {e}")


    convo_frame = None
    convo_enabled = CONFIG.ENABLE_CONVERSATION
    convo_spec = {"start": None, "questions": {}}
    q_map = {}
    current_qid = [None]
    answers = {}


    confirm_mode_active = False
    blink_status_var = tk.StringVar(value="Blink confirm: idle")
    confirm_frame = ttk.LabelFrame(sidebar, text="Blink Confirmation")
    confirm_frame.pack(fill=tk.X, padx=0, pady=(0,8))
    ttk.Label(confirm_frame, textvariable=blink_status_var, wraplength=260, justify="left").pack(fill=tk.X, padx=6, pady=(6,6))


    latest_frame_rgb = None
    latest_frame_lock = threading.Lock()
    blink_lock = threading.Lock()

    def build_convo_panel():
        nonlocal convo_frame, q_map, current_qid, answers
        if convo_frame is not None or not convo_enabled:
            return

        convo_frame = ttk.LabelFrame(sidebar, text="Conversation (EMG)")
        convo_frame.pack(fill=tk.X, padx=0, pady=(0,8))

        q_text_var = tk.StringVar(value="Waiting for conversation spec…")
        ttk.Label(convo_frame, textvariable=q_text_var, wraplength=260, justify="left").pack(fill=tk.X, padx=6, pady=(6,6))

        pb = ttk.Progressbar(convo_frame, orient="horizontal", mode="determinate")
        pb.pack(fill=tk.X, padx=6, pady=(0,4))
        pb["maximum"] = int(CONFIG.VOICE_CAPTURE_SEC * 1000); pb["value"] = 0

        listen_var = tk.StringVar(value="")
        ttk.Label(convo_frame, textvariable=listen_var, justify="left").pack(fill=tk.X, padx=6, pady=(0,4))

        transcript = tk.Text(convo_frame, height=6, width=36)
        transcript.pack(fill=tk.BOTH, padx=6, pady=(0,6))

        convo_frame.q_text_var = q_text_var
        convo_frame.pb = pb
        convo_frame.listen_var = listen_var
        convo_frame.transcript = transcript

        def tlog(s: str):
            transcript.insert(tk.END, s + "\n"); transcript.see(tk.END)
        def tclear():
            transcript.delete("1.0","end")
        convo_frame.tlog = tlog; convo_frame.tclear = tclear

        def followup_for(q: dict, val):
            fups = q.get("followups", {}) or {}
            def first_branch(d: dict): return next(iter(d.values()), None) if d else None
            key = (val or "").lower() if isinstance(val, str) else val
            return fups.get(key, fups.get("_any", first_branch(fups)))
        convo_frame.followup_for = followup_for

        capture_start_ts = [0.0]
        def start_capture_for(qid: Optional[str]):
            if not qid:
                q_text_var.set("All set. Thank you."); listen_var.set(""); pb["value"] = 0
                speak("All set. Thank you.", flush=True)
                return
            q = q_map.get(qid)
            if not q:
                q_text_var.set("All set. Thank you."); listen_var.set(""); pb["value"] = 0
                speak("All set. Thank you.", flush=True)
                return

            text = q.get("text", "").strip()
            q_text_var.set(text)
            tclear()
            listen_var.set(f"Listening {CONFIG.VOICE_CAPTURE_SEC:.1f}s (EMG)…")
            pb["value"] = 0; capture_start_ts[0] = time.time()


            speak(text, flush=True)

            meta = {"qid": qid, "qtype": q.get("type","choice")}
            if meta["qtype"] in ("choice", "multi"):
                meta["options"] = q.get("options", [])
            if meta["qtype"] == "scale":
                meta["min"] = int(q.get("min", 0)); meta["max"] = int(q.get("max", 10)); meta["step"] = int(q.get("step", 1))
            if meta["qtype"] == "text":
                meta["placeholder"] = q.get("placeholder", "")
            try: from_gui.put({"type":"start_voice_capture", **meta})
            except Exception: pass

            def tick():
                elapsed_ms = int((time.time() - capture_start_ts[0]) * 1000)
                pb["value"] = min(pb["maximum"], max(0, elapsed_ms))
                if elapsed_ms < pb["maximum"]:
                    root.after(50, tick)
            root.after(50, tick)
        convo_frame.start_capture_for = start_capture_for


        REDUNDANCY_RULES = [
            {"qid": "q_alert", "next_is": None, "answers": {"pain","yes"}, "force_next": None},
            {"qid": "q_fever", "next_is": "q_pain_relief", "if_answers": {"q_pain_now": {"no","hmmm"}}, "force_next": "q_rest"},
            {"qid": "q_rest", "next_is": "q_alert",
             "if_answers_all": {"q_pain_now": {"no","hmmm"}, "q_breathing": {"no","hmmm"}, "q_dizzy": {"no","hmmm"}, "q_fever": {"no","hmmm"}},
             "force_next": "q_followup"},
        ]

        def apply_redundancy_rules(qid: str, val: str, nxt: Optional[str]) -> Optional[str]:
            v = (val or "").lower()
            for rule in REDUNDANCY_RULES:
                if qid != rule["qid"]:
                    continue
                if "next_is" in rule and rule["next_is"] is not None and nxt != rule["next_is"]:
                    continue
                if "answers" in rule and v not in rule["answers"]:
                    continue
                ok = True
                if "if_answers" in rule:
                    for rqid, allowed in rule["if_answers"].items():
                        if answers.get(rqid) not in allowed:
                            ok = False; break
                if ok and "if_answers_all" in rule:
                    for rqid, allowed in rule["if_answers_all"].items():
                        if answers.get(rqid) not in allowed:
                            ok = False; break
                if not ok:
                    continue
                return rule["force_next"]
            return nxt

        def on_voice_decision(payload: dict):
            qid = payload.get("qid"); q = q_map.get(qid, {})
            val = payload.get("decision"); conf = payload.get("confidence", None)
            if conf is None: convo_frame.tlog(f"[Decision] {qid}: {val}")
            else:            convo_frame.tlog(f"[Decision] {qid}: {val} (conf={conf:.2f})")
            answers[qid] = (val or "").lower()
            nxt = convo_frame.followup_for(q, val)
            nxt = apply_redundancy_rules(qid, str(val), nxt)
            current_qid[0] = nxt
            convo_frame.tlog(f"→ Next: {nxt if nxt is not None else 'END'}")
            convo_frame.start_capture_for(nxt)
        convo_frame.on_voice_decision = on_voice_decision

    def set_convo_spec(spec: dict):
        nonlocal convo_spec, q_map, current_qid, answers
        build_convo_panel()
        convo_spec = spec or {"start": None, "questions": {}}
        q_map.clear(); q_map.update(convo_spec.get("questions", {}) or {})
        current_qid[0] = convo_spec.get("start"); answers.clear()
        if convo_frame and current_qid[0]:
            convo_frame.start_capture_for(current_qid[0])


    fs = None; live_sec = 5.0
    sig_ring = None; sig_wptr = 0; sig_filled = 0; sig_lines = []; n_ch_live = 0; max_window_sec = 5.0
    paused_live = False
    def setup_signal(n_ch: int, fs_in: float, live_sec_in: float):
        nonlocal fs, live_sec, sig_ring, sig_wptr, sig_filled, sig_lines, n_ch_live, max_window_sec
        fs = float(fs_in); live_sec = float(live_sec_in)
        n_ch_live = int(n_ch); max_window_sec = live_sec
        Tbuf = max(1, int(round(live_sec * fs)))
        sig_ring = np.zeros((n_ch_live, Tbuf), dtype=np.float32)
        sig_wptr = 0; sig_filled = 0; sig_lines = []
        ax_sig.clear(); ax_sig.set_title("Live signal (gate channels) — stacked, z-norm per channel")
        ax_sig.set_ylabel("Amplitude (a.u.)"); ax_sig.grid(True, axis="y", alpha=0.25); ax_sig.margins(y=0.10)
        for i in range(n_ch_live):
            ln, = ax_sig.plot([], [], linewidth=1.0, label=f"CH{i+1}"); sig_lines.append(ln)
        if n_ch_live <= 12: ax_sig.legend(loc="upper right", fontsize=8)
        canvas_sig.draw_idle()
        logln(f"[INIT] Signal plot: {n_ch_live} channels @ {fs:.2f} Hz; buffer={Tbuf} (~{Tbuf/fs:.2f}s)")
    def update_prob(names: List[str], probs: List[float]):
        ax_prob.clear()
        ax_prob.set_title("Last inference — class probability distribution")
        ax_prob.set_ylabel("Probability")
        ax_prob.grid(True, axis="y", alpha=0.25)
        x = np.arange(len(probs))
        ax_prob.bar(x, probs); ax_prob.set_xticks(x)
        ax_prob.set_xticklabels([n.upper() for n in names], rotation=45, ha="right")
        max_p = float(np.max(probs)) if len(probs) else 0.0
        top = max(0.1, max_p * 1.10); ax_prob.set_ylim(0.0, min(1.0, top))
        canvas_prob.draw_idle()
    def append_signal(chunk_cxn: np.ndarray):
        nonlocal sig_ring, sig_wptr, sig_filled
        if paused_live or sig_ring is None or chunk_cxn.size == 0: return
        C, N = chunk_cxn.shape; Tbuf = sig_ring.shape[1]; N = int(N)
        if N >= Tbuf:
            sig_ring[:] = chunk_cxn[:, -Tbuf:]; sig_wptr = 0; sig_filled = Tbuf
        else:
            end = sig_wptr + N
            if end <= Tbuf:
                sig_ring[:, sig_wptr:end] = chunk_cxn
            else:
                part = Tbuf - sig_wptr
                sig_ring[:, sig_wptr:] = chunk_cxn[:, :part]
                sig_ring[:, :N-part] = chunk_cxn[:, part:]
            sig_wptr = (sig_wptr + N) % Tbuf
            sig_filled = min(Tbuf, sig_filled + N)
        if sig_filled > 0 and fs and fs > 0:
            view = sig_ring if sig_wptr == 0 else np.concatenate([sig_ring[:, sig_wptr:], sig_ring[:, :sig_wptr]], axis=1)
            Tcur = int(sig_filled)
            x = np.linspace(-Tcur / float(fs), 0.0, Tcur, dtype=np.float32)
            vis_left = -min(max_window_sec, Tcur / float(fs)); ax_sig.set_xlim([vis_left, 0.0])
            for i in range(view.shape[0]):
                v = view[i, -Tcur:]; sdv = float(np.std(v)) if np.std(v) > 1e-6 else 1.0
                y = (v / sdv) + (i * 3.0); sig_lines[i].set_data(x, y)
            ax_sig.relim(); ax_sig.autoscale_view(scalex=False, scaley=True); canvas_sig.draw_idle()
    def clear_last_window_axes():
        ax_last.clear(); ax_last.set_title("Last inference window (model channels)")
        ax_last.set_xlabel("Time (s)"); ax_last.set_ylabel("Amplitude (a.u., stacked)")
        ax_last.grid(True, axis="y", alpha=0.25); ax_last.margins(y=0.10); canvas_last.draw_idle()
    def update_last_window(chunk_cxn: np.ndarray, fs_win: float, trig_idx: int, ch_labels: Optional[List[str]] = None):
        C, T = chunk_cxn.shape; fsw = float(fs_win) if fs_win else 250.0
        t = (np.arange(T, dtype=np.float32) - float(trig_idx)) / fsw
        ax_last.clear()
        ax_last.set_title("Last inference window (model channels)")
        ax_last.set_xlabel("Time (s)"); ax_last.set_ylabel("Amplitude (a.u., stacked)")
        ax_last.grid(True, axis="y", alpha=0.25); ax_last.margins(y=0.10)
        for i in range(C):
            v = chunk_cxn[i]; sdv = float(np.std(v)) if np.std(v) > 1e-6 else 1.0
            y = (v / sdv) + (i * 3.0); label = ch_labels[i] if (ch_labels and i < len(ch_labels)) else f"CH{i+1}"
            ax_last.plot(t, y, linewidth=1.0, label=label)
        ax_last.axvline(0.0, linestyle="--", linewidth=1.2); ax_last.set_xlim([float(t[0]), float(t[-1])])
        ax_last.relim(); ax_last.autoscale_view(scalex=False, scaley=True)
        if C <= 12: ax_last.legend(loc="upper right", fontsize=8)
        canvas_last.draw_idle()
    clear_last_window_axes()


    try:
        root.update_idletasks()
        from_gui.put({"type": "gui_ready"})
    except Exception:
        pass


    def camera_loop():
        nonlocal latest_frame_rgb, confirm_mode_active
        if not _HAS_CV2:
            print("[GUI] Camera preview disabled (cv2 missing).")
            return

        cap = None
        face_mesh = None
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("[GUI] Could not open webcam.")
                return

            if _HAS_MEDIAPIPE:
                mp_face_mesh = mp.solutions.face_mesh
                face_mesh = mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                LEFT_EYE_OUTER = 33;  LEFT_EYE_INNER = 133; LEFT_EYE_TOP = 159; LEFT_EYE_BOTTOM = 145
                RIGHT_EYE_OUTER = 362; RIGHT_EYE_INNER = 263; RIGHT_EYE_TOP = 386; RIGHT_EYE_BOTTOM = 374
                BLINK_EAR_THRESH = 0.20
                BLINK_CONSEC_FRAMES = 3
                blink_counter = 0

                def _dist(p1, p2): return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) ** 0.5
                def _ear(pts):
                    hor = _dist(pts['outer'], pts['inner']) + 1e-6
                    ver = _dist(pts['top'], pts['bottom'])
                    return ver / hor
                def _get_eye_points(landmarks, w, h):
                    def lm(idx):
                        p = landmarks[idx]
                        return (p.x * w, p.y * h)
                    left = {"outer": lm(LEFT_EYE_OUTER), "inner": lm(LEFT_EYE_INNER), "top": lm(LEFT_EYE_TOP), "bottom": lm(LEFT_EYE_BOTTOM)}
                    right= {"outer": lm(RIGHT_EYE_OUTER),"inner": lm(RIGHT_EYE_INNER),"top": lm(RIGHT_EYE_TOP), "bottom": lm(RIGHT_EYE_BOTTOM)}
                    return left, right

            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    _time.sleep(0.05)
                    continue

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                if face_mesh is not None:
                    h, w = frame_rgb.shape[:2]
                    results = face_mesh.process(frame_rgb)
                    if results.multi_face_landmarks:
                        face = results.multi_face_landmarks[0].landmark
                        left_eye, right_eye = _get_eye_points(face, w, h)
                        ear_left = _ear(left_eye); ear_right = _ear(right_eye)
                        ear = (ear_left + ear_right) / 2.0
                        if ear < BLINK_EAR_THRESH:
                            blink_counter += 1
                        else:
                            if blink_counter >= BLINK_CONSEC_FRAMES:
                                with blink_lock:
                                    active = confirm_mode_active
                                if active:
                                    from_gui.put({"type": "blink_confirm"})
                                    _time.sleep(0.3)
                            blink_counter = 0

                with latest_frame_lock:
                    latest_frame_rgb = frame_rgb

                _time.sleep(0.01)

        finally:
            if face_mesh is not None:
                face_mesh.close()
            if cap is not None:
                cap.release()

    threading.Thread(target=camera_loop, daemon=True).start()

    def update_camera_view():
        if not (_HAS_PIL and _HAS_CV2):
            root.after(100, update_camera_view); return
        frame = None
        with latest_frame_lock:
            if latest_frame_rgb is not None:
                frame = latest_frame_rgb.copy()
        if frame is not None:
            h, w = frame.shape[:2]
            target_w = CAM_PREVIEW_WIDTH
            target_h = max(1, int(h * (target_w / float(w))))
            try:
                from PIL import Image, ImageTk
                img = Image.fromarray(frame).resize((target_w, target_h), Image.BILINEAR)
                photo = ImageTk.PhotoImage(img)
                cam_label.configure(image=photo, text="")
                cam_label._photo = photo
            except Exception:
                pass
        root.after(33, update_camera_view)
    root.after(100, update_camera_view)


    def poll_queue():
        nonlocal confirm_mode_active
        try:
            while True:
                msg = to_gui.get_nowait()
                mtyp = msg.get("type", "")
                if mtyp == "init":
                    names = msg.get("class_names", [])
                    update_prob(names, [0.0] * len(names))
                    info_var.set(msg.get("info", ""))
                    if "fs" in msg and "live_sec" in msg and "n_signal_ch" in msg:
                        setup_signal(int(msg["n_signal_ch"]), float(msg["fs"]), float(msg["live_sec"]))
                elif mtyp == "signal":
                    ch = np.array(msg.get("chunk", []), dtype=np.float32)
                    append_signal(ch)
                elif mtyp == "update":
                    names = msg.get("class_names", [])
                    probs = msg.get("probs", [])
                    top1  = msg.get("top1", ("", 0.0))
                    status_var.set(f"{msg.get('timestamp','--:--:--')}  |  Top-1: {top1[0].upper()}  ({top1[1]*100:.1f}%)")
                    info_var.set(msg.get("info",""))
                    update_prob(names, probs)
                    if "logline" in msg: logln(msg["logline"])
                elif mtyp == "last_window":
                    chunk = np.array(msg.get("chunk", []), dtype=np.float32)
                    fs_win = float(msg.get("fs", 250.0))
                    trig_idx = int(msg.get("trigger_idx", 0))
                    labels = msg.get("labels", None)
                    update_last_window(chunk, fs_win, trig_idx, labels)
                elif mtyp == "gate_info":
                    try:
                        gate_mode = "uv" if bool(msg.get("use_uv", False)) else "z"
                        gate_mode_var.set(gate_mode)
                        uv_threshold_var.set(float(msg.get("threshold_uV", uv_threshold_var.get())))
                        z_threshold_var.set(float(msg.get("zscore_thresh", z_threshold_var.get())))
                        min_active_var.set(int(msg.get("min_active_channels", min_active_var.get())))
                        consec_var.set(int(msg.get("consec_samp", consec_var.get())))
                        cooldown_var.set(int(msg.get("cooldown_samp", cooldown_var.get())))
                        b = str(msg.get("board", "SYNTHETIC")).upper()
                        if b in ("CYTON", "SYNTHETIC"): board_var.set(b)
                    except Exception as e:
                        logln(f"[INIT] Gate settings sync failed: {e}")
                elif mtyp == "convo_spec":
                    spec = msg.get("spec", {"start": None, "questions": {}})
                    set_convo_spec(spec)
                elif mtyp == "voice_capture_step":
                    token = msg.get("token",""); conf = msg.get("conf", None)
                    if convo_frame is not None:
                        if conf is None: convo_frame.tlog(f"{token}")
                        else:            convo_frame.tlog(f"{token} ({conf:.2f})")
                elif mtyp == "voice_decision":
                    if convo_frame is not None:
                        convo_frame.on_voice_decision(msg)
                elif mtyp == "confirm_prompt":
                    word = msg.get("word", "")
                    conf = msg.get("confidence", 0.0)
                    blink_status_var.set(
                        f"Predicted: {word.upper()} ({conf*100:.1f}%).\nBlink within 2 seconds to confirm."
                    )
                    with blink_lock:
                        confirm_mode_active = True
                elif mtyp == "confirm_prompt_clear":
                    blink_status_var.set("Blink confirm: idle")
                    with blink_lock:
                        confirm_mode_active = False
                elif mtyp == "log":
                    logln(msg.get("text",""))
                elif mtyp == "stop":
                    root.after(50, root.destroy); return
        except queue.Empty:
            pass
        root.after(poll_ms, poll_queue)

    def on_close():
        try: from_gui.put({"type": "quit"})
        except Exception: pass
        try:
            if _HAS_TTS_BACKEND:
                tts_queue.put(None)
            if _HAS_SD:
                sd.stop()
        except Exception:
            pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(poll_ms, poll_queue)
    root.mainloop()





def stack_features(arr: np.ndarray, sfreq: float, env_ms=20, rms_ms=20) -> np.ndarray:
    win_env = max(3, int((env_ms / 1000.0) * sfreq))
    env = uniform_filter1d(np.abs(arr), size=win_env, axis=1, mode="reflect")
    win_rms = max(3, int((rms_ms / 1000.0) * sfreq))
    ms = uniform_filter1d(arr * arr, size=win_rms, axis=1, mode="reflect")
    rms = np.sqrt(ms + 1e-12)
    return np.concatenate([arr, env, rms], axis=0).astype(np.float32)


def build_atcnet(n_chans, n_classes, n_times, sfreq):
    secs = float(n_times) / float(sfreq)
    candidates = [
        dict(n_chans=n_chans, n_outputs=n_classes, n_times=n_times,
             input_window_seconds=secs, sfreq=sfreq, n_windows=3,
             att_head_dim=8, att_num_heads=2, tcn_depth=2, tcn_kernel_size=4,
             conv_block_n_filters=16, conv_block_kernel_length_1=64, conv_block_kernel_length_2=16,
             conv_block_pool_size_1=8, conv_block_pool_size_2=7, conv_block_depth_mult=2,
             conv_block_dropout=0.30, concat=False),
        dict(n_chans=n_chans, n_outputs=n_classes, n_times=n_times,
             input_window_seconds=None, sfreq=sfreq, n_windows=3,
             att_head_dim=8, att_num_heads=2, tcn_depth=2, tcn_kernel_size=4,
             conv_block_n_filters=16, conv_block_kernel_length_1=64, conv_block_kernel_length_2=16,
             conv_block_pool_size_1=8, conv_block_pool_size_2=7, conv_block_depth_mult=2,
             conv_block_dropout=0.30, concat=False),
    ]
    last_err = None
    for kw in candidates:
        try: return ATCNet(**kw)
        except Exception as e: last_err = e
    raise TypeError(f"Could not instantiate ATCNet; last error: {last_err}")


def load_ckpt(path, n_chans_model, n_classes, n_times, sfreq, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = build_atcnet(n_chans_model, n_classes, n_times, sfreq).to(device)
    state = ckpt.get("state_dict", ckpt)
    clean_state = {}
    for k, v in state.items():
        if k == "n_averaged" or k.endswith(".n_averaged"): continue
        clean_state[k.replace("module.", "")] = v
    model.load_state_dict(clean_state, strict=False)
    model.eval()
    meta = {"class_names": ckpt.get("class_names", None),
            "sfreq": ckpt.get("sfreq", None),
            "target_len": ckpt.get("target_len", None)}
    return model, meta


class EMGClassifier:
    def __init__(self, ckpt_path: str, top_channels_1based: List[int],
                 class_names_hint: Optional[List[str]] = None,
                 sfreq_hint: Optional[float] = None,
                 target_len_hint: Optional[int] = None):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        meta0 = torch.load(ckpt_path, map_location="cpu")
        meta_classes = meta0.get("class_names", None)
        meta_sfreq   = meta0.get("sfreq", None)
        meta_tlen    = meta0.get("target_len", None)

        self.class_names = [str(s).lower() for s in (meta_classes or class_names_hint or [])]
        self.sfreq = float(meta_sfreq) if meta_sfreq is not None else (float(sfreq_hint) if sfreq_hint is not None else 250.0)
        self.target_len = int(meta_tlen) if meta_tlen is not None else (int(target_len_hint) if target_len_hint is not None else int(2.0 * self.sfreq))

        self.sel_channels = sorted([int(c)-1 for c in top_channels_1based])
        if len(self.sel_channels) == 0:
            raise ValueError("No MODEL_CHANNELS provided. Set CONFIG.USED_CHANNELS (or MODEL_CHANNELS).")

        n_classes_guess = len(self.class_names) if self.class_names else 5
        self.n_chans_model = 3 * len(self.sel_channels)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, meta = load_ckpt(ckpt_path, self.n_chans_model, n_classes_guess,
                                self.target_len, self.sfreq, device=str(self.device))

        if meta.get("class_names") is not None:
            self.class_names = [str(s).lower() for s in meta["class_names"]]
        if meta.get("sfreq") is not None:
            self.sfreq = float(meta["sfreq"])
        if meta.get("target_len") is not None:
            self.target_len = int(meta["target_len"])

        self.model = model
        self.idx_map = {name.lower(): i for i, name in enumerate(self.class_names)}

        if not self.class_names:
            out_features = getattr(self.model, 'final_layer', None).out_features if hasattr(self.model,'final_layer') else n_classes_guess
            self.class_names = [f"class_{i}" for i in range(out_features)]

    def _prepare(self, emg_CxT: np.ndarray) -> torch.Tensor:
        a = np.asarray(emg_CxT, dtype=np.float32)
        assert a.ndim == 2, f"Expected (C, T), got {a.shape}"
        a = a[self.sel_channels]
        a = stack_features(a, self.sfreq)
        assert a.shape[0] == self.n_chans_model, f"Stacked channel dim mismatch: got {a.shape[0]}, expected {self.n_chans_model}"
        return torch.from_numpy(a).unsqueeze(0).to(self.device)

    def predict_proba(self, emg_CxT: np.ndarray) -> Tuple[np.ndarray, float]:
        x = self._prepare(emg_CxT)
        t0 = time.time()
        with torch.no_grad():
            out = self.model(x).view(x.size(0), -1)
            p = F.softmax(out, dim=1).cpu().numpy()[0]
        dt = (time.time() - t0)
        return p, dt

    def topk(self, p: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        k = min(k, len(p))
        idx = np.argsort(-p)[:k].tolist()
        return [(self.class_names[i] if i < len(self.class_names) else f"class_{i}", float(p[i])) for i in idx]





def butter_highpass(sig: np.ndarray, fs: float, cutoff: float = 15.0, order: int = 4) -> np.ndarray:
    if cutoff <= 0: return sig
    ny = 0.5 * fs; Wn = cutoff / ny
    b, a = butter(order, Wn, btype='highpass')
    return filtfilt(b, a, sig, axis=1)

def notch(sig: np.ndarray, fs: float, freq: float = 60.0, Q: float = 30.0) -> np.ndarray:
    if freq <= 0: return sig
    b, a = iirnotch(w0=freq/(fs/2.0), Q=Q)
    return filtfilt(b, a, sig, axis=1)

def zscore(sig: np.ndarray, axis=1, eps=1e-6) -> np.ndarray:
    m = sig.mean(axis=axis, keepdims=True)
    s = sig.std(axis=axis, keepdims=True)
    return (sig - m) / (s + eps)





@dataclass
class GateConfig:
    threshold_uV: float = 50.0
    min_active_channels: int = 2
    pre_samp: int = 800
    post_samp: int = 1200
    refractory_samp: int = 600
    use_uv_gate: bool = True
    zscore_thresh: float = 4.5
    consec_samp: int = 1

def detect_gate_indices(chunk_vals: np.ndarray, cfg: GateConfig) -> List[int]:
    C, N = chunk_vals.shape
    if N == 0: return []
    M = max(1, int(cfg.consec_samp))
    if N < M: return []
    thr = cfg.threshold_uV if cfg.use_uv_gate else cfg.zscore_thresh
    above = (np.abs(chunk_vals) >= thr).astype(np.int16)
    kern = np.ones(M, dtype=np.int16)
    runs = np.zeros((C, N - M + 1), dtype=np.int16)
    for c in range(C):
        runs[c] = np.convolve(above[c], kern, mode='valid')
    chan_ok = (runs == M).sum(axis=0)
    idx = np.where(chan_ok >= cfg.min_active_channels)[0]
    if idx.size == 0: return []
    return [int(idx[0] + (M - 1))]





NUMBER_WORDS = {"zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5","six":"6","seven":"7","eight":"8","nine":"9","ten":"10"}
VOICE_SYNONYMS = {
    "yeah":"yes","yep":"yes","yup":"yes","sure":"yes","affirmative":"yes","ok":"yes","okay":"yes",
    "nope":"no","nah":"no","negative":"no",
    "hm":"hmmm","hmm":"hmmm","hmmm":"hmmm","umm":"hmmm","uh":"hmmm",
    "pain?":"pain","painful":"pain","ache":"pain","ow":"pain"
}
def norm_token(s: str) -> str:
    s = s.replace("_"," ").lower().strip()
    s = "".join(ch if ch.isalnum() or ch==" " else " " for ch in s)
    s = " ".join(s.split())
    s = NUMBER_WORDS.get(s, s)
    s = VOICE_SYNONYMS.get(s, s)
    return s
def build_vocab(class_names: List[str]) -> set:
    return { norm_token(c) for c in class_names }
def restrict_spec_to_allowed(spec_in: dict, allowed: List[str], model_vocab: set) -> dict:
    allowed_norm = [norm_token(a) for a in allowed]
    present = [a for a in allowed_norm if a in model_vocab]
    options_final = present if present else allowed_norm
    spec = copy.deepcopy(spec_in)
    qmap = spec.get("questions", {}) or {}
    for qid, q in qmap.items():
        q["type"] = "choice"
        q["options"] = options_final[:]
        f = q.get("followups", {}) or {}
        default_next = f.get("_any")
        if default_next is None and f: default_next = next(iter(f.values()))
        if default_next is None: default_next = None
        f_out = {"_any": default_next}
        for opt in q["options"]:
            f_out[opt] = f.get(opt, default_next)
        q["followups"] = f_out
    return spec





def main():
    if not os.path.exists(CONFIG.CKPT_PATH):
        raise FileNotFoundError(f"Edit CONFIG.CKPT_PATH to point to your checkpoint. Not found: {CONFIG.CKPT_PATH}")

    to_gui = Queue(); from_gui = Queue()
    gp = Process(target=gui_process, args=(to_gui, from_gui, CONFIG.GUI_TITLE, CONFIG.GUI_POLL_MS), daemon=True)
    gp.start()


    gui_ready = False
    t0 = time.time()
    while (time.time() - t0) < 10.0:
        try:
            msg = from_gui.get(timeout=0.2)
            if msg.get("type") == "gui_ready":
                gui_ready = True; break
            else:
                from_gui.put(msg)
        except queue.Empty:
            pass
    if not gui_ready:
        print("[Init] GUI ready signal not received within timeout; continuing anyway.", flush=True)

    current_board_name = str(CONFIG.BOARD_MODE).upper().strip()
    live = LiveBoard(serial_port=CONFIG.SERIAL, board_id=current_board_name)
    fs_used = float(CONFIG.FS_OVERRIDE) if CONFIG.FS_OVERRIDE is not None else float(live.fs)

    sel_gate = [c-1 for c in CONFIG.GATE_CHANNELS]
    sel_gate = [s for s in sel_gate if 0 <= s < live.n_ch]
    if not sel_gate: sel_gate = list(range(min(4, live.n_ch)))

    to_gui.put({"type": "log", "text": f"Starting board: {current_board_name} @ {live.fs} Hz"})
    to_gui.put({"type": "log", "text": f"Streaming channels (row idx): {live.ch_idx}"})
    to_gui.put({"type": "log", "text": f"Gate channels (0-based rows): {sel_gate}"})
    to_gui.put({"type": "log", "text": f"USED_CHANNELS={CONFIG.USED_CHANNELS}  MODEL={CONFIG.MODEL_CHANNELS}  GATE={CONFIG.GATE_CHANNELS}"})


    clf = EMGClassifier(
        ckpt_path=CONFIG.CKPT_PATH,
        top_channels_1based=CONFIG.MODEL_CHANNELS,
        class_names_hint=None,
        sfreq_hint=fs_used,
        target_len_hint=None
    )


    if CONFIG.ENABLE_CONVERSATION:
        model_vocab = build_vocab(clf.class_names)
        conv_spec_restricted = restrict_spec_to_allowed(CONFIG.CONVERSATION_SPEC, CONFIG.ALLOWED_RESPONSES, model_vocab)
        to_gui.put({"type":"convo_spec","spec": conv_spec_restricted})


    try:
        to_gui.put({"type": "log", "text": "Warming up model…"})
        _zeros = np.zeros((len(CONFIG.MODEL_CHANNELS), max(8, clf.target_len)), dtype=np.float32)
        _ = clf.predict_proba(_zeros)
        to_gui.put({"type": "log", "text": "Model warm-up complete."})
    except Exception as e:
        to_gui.put({"type": "log", "text": f"[Warm-up skipped] {e}"})


    to_gui.put({
        "type": "init",
        "class_names": clf.class_names,
        "info": f"Classes={len(clf.class_names)} | Pre={CONFIG.PRE_SAMP} Post={CONFIG.POST_SAMP} (total={CONFIG.PRE_SAMP+CONFIG.POST_SAMP})",
        "fs": fs_used,
        "live_sec": CONFIG.LIVE_PLOT_SEC,
        "n_signal_ch": len(sel_gate)
    })


    pre_samp = int(CONFIG.PRE_SAMP); post_samp = int(CONFIG.POST_SAMP)
    total = pre_samp + post_samp
    refractory_samp = int(CONFIG.REFRACTORY_SAMP) if CONFIG.REFRACTORY_SAMP is not None else max(1, post_samp // 2)
    cooldown_samp = int(CONFIG.COOLDOWN_SAMP)

    gate_cfg = GateConfig(
        threshold_uV=50.0,
        min_active_channels=CONFIG.MIN_ACTIVE_CHANNELS,
        pre_samp=pre_samp, post_samp=post_samp, refractory_samp=refractory_samp,
        use_uv_gate=(CONFIG.USE_UV_GATE and (current_board_name != "SYNTHETIC")),
        zscore_thresh=CONFIG.ZSCORE_THRESH, consec_samp=CONFIG.GATE_CONSEC_SAMP
    )

    if current_board_name == "SYNTHETIC":
        gate_cfg.use_uv_gate = False
        gate_cfg.zscore_thresh = CONFIG.SYNTHETIC_Z_THRESH
        gate_cfg.min_active_channels = CONFIG.SYNTHETIC_MIN_ACTIVE
        gate_cfg.consec_samp = CONFIG.SYNTHETIC_CONSEC
        to_gui.put({"type": "log", "text": f"[Synthetic gate] |z| ≥ {gate_cfg.zscore_thresh}, min_channels={gate_cfg.min_active_channels}, consecutive={gate_cfg.consec_samp}"})
    else:
        to_gui.put({"type": "log", "text": f"[Real gate] mode={'µV' if gate_cfg.use_uv_gate else 'z-score'}, thr={gate_cfg.threshold_uV if gate_cfg.use_uv_gate else gate_cfg.zscore_thresh}, min_channels={gate_cfg.min_active_channels}, consecutive={gate_cfg.consec_samp}"})

    to_gui.put({"type": "log", "text": f"Window: pre={pre_samp}, post={post_samp}, total={total}. Refractory={refractory_samp}. Cooldown={cooldown_samp}."})


    to_gui.put({
        "type": "gate_info",
        "use_uv": gate_cfg.use_uv_gate,
        "threshold_uV": gate_cfg.threshold_uV,
        "zscore_thresh": gate_cfg.zscore_thresh,
        "min_active_channels": gate_cfg.min_active_channels,
        "consec_samp": gate_cfg.consec_samp,
        "cooldown_samp": cooldown_samp,
        "board": current_board_name
    })


    ring_len = max((pre_samp + post_samp) * CONFIG.RING_FACTOR, pre_samp + post_samp + 1000)
    ring = np.zeros((live.n_ch, ring_len), dtype=np.float32)
    wptr = 0; filled = 0

    sel_model = [c-1 for c in CONFIG.MODEL_CHANNELS]
    model_labels = [f"CH{c}" for c in CONFIG.MODEL_CHANNELS]

    last_trigger_wptr = -10_000_000
    samples_since_last_trigger = cooldown_samp + 1
    last_gate_walltime = time.time()
    use_uv = gate_cfg.use_uv_gate
    stop = False
    trigger_id = 0
    empty_reads = 0

    print("[Run] Streaming. (Close GUI window or Ctrl+C to exit)")

    def have_tail_window(T_needed: int) -> bool:
        return filled >= T_needed
    def get_tail_window(T_needed: int) -> Optional[np.ndarray]:
        if filled < T_needed: return None
        start = (wptr - T_needed) % ring_len
        end_  = wptr % ring_len
        if start < end_:
            return ring[:, start:end_].copy()
        else:
            return np.concatenate([ring[:, start:], ring[:, :end_]], axis=1).copy()

    voice_session = {
        "active": False, "qid": None, "qtype": None,
        "options": [], "options_norm": [],
        "min": 0, "max": 10,
        "end_time": 0.0, "next_step": 0.0,
        "counts": {}, "numbers": [], "texts": [],
        "steps": 0,
    }

    URGENT_TOKENS = {"pain"}
    CONFIRM_THRESHOLD = 0.70
    pending_confirmation = {"active": False, "qid": None, "qtype": None, "decision": "", "confidence": 0.0, "start_ts": 0.0}

    def send_voice_decision(qid, qtype, decision, confidence):
        to_gui.put({"type": "voice_decision", "qid": qid, "qtype": qtype, "decision": decision, "confidence": confidence})
        to_gui.put({"type": "log","text": f"[Voice] Final decision {qid} → {decision} (conf≈{confidence:.2f})"})

    def maybe_require_confirmation(qid, qtype, decision, confidence):
        nonlocal pending_confirmation
        if not decision:
            send_voice_decision(qid, qtype, "hmmm", 0.0); return
        token = (decision or "").lower()
        is_urgent = token in URGENT_TOKENS
        if is_urgent or confidence >= CONFIRM_THRESHOLD:
            send_voice_decision(qid, qtype, decision, confidence); return
        pending_confirmation = {"active": True, "qid": qid, "qtype": qtype, "decision": decision, "confidence": confidence, "start_ts": time.time()}
        to_gui.put({"type": "confirm_prompt", "word": decision, "confidence": confidence})
        to_gui.put({"type": "log", "text": (f"[Confirm] Underconfident non-urgent '{decision}' "
                                            f"(conf≈{confidence:.2f}). Waiting for blink (2s)…")})

    def finish_voice_session(session: dict):
        qid = session["qid"]; qtype = session["qtype"]
        decision = ""; confidence = 0.0
        counts = session["counts"]
        if counts:
            sorted_items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            decision, c = sorted_items[0]
            total = sum(counts.values())
            confidence = c / total if total > 0 else 0.0
        session["active"] = False
        maybe_require_confirmation(qid, qtype, decision, confidence)

    def start_voice_session(meta: dict):
        voice_session["active"] = True
        voice_session["qid"] = meta.get("qid")
        voice_session["qtype"] = meta.get("qtype")
        opts = meta.get("options", [])
        voice_session["options"] = list(opts)
        voice_session["options_norm"] = [norm_token(o) for o in opts]
        voice_session["min"] = int(meta.get("min", 0))
        voice_session["max"] = int(meta.get("max", 10))
        now = time.time()
        voice_session["end_time"] = now + CONFIG.VOICE_CAPTURE_SEC
        voice_session["next_step"] = now
        voice_session["counts"] = {}
        voice_session["numbers"] = []
        voice_session["texts"] = []
        voice_session["steps"] = 0
        to_gui.put({"type":"log","text":f"[Voice] Capture start {voice_session['qid']} for {CONFIG.VOICE_CAPTURE_SEC:.1f}s"})

    def voice_step_infer(session: dict):
        T_target = clf.target_len
        X = get_tail_window(T_target)
        if X is None: return
        if CONFIG.HP_CUTOFF > 0: X = butter_highpass(X, fs=fs_used, cutoff=float(CONFIG.HP_CUTOFF), order=4)
        if CONFIG.NOTCH_HZ in (50, 60): X = notch(X, fs=fs_used, freq=float(CONFIG.NOTCH_HZ), Q=30.0)
        X = zscore(X, axis=1)
        if X.shape[1] != T_target: X = resample(X, T_target, axis=1)
        try:
            p, _ = clf.predict_proba(X)
        except Exception as e:
            to_gui.put({"type":"log","text":f"[Voice] step infer error: {e}"}); return
        top_items = clf.topk(p, k=1)
        token_raw = str(top_items[0][0]).strip().lower()
        token = norm_token(token_raw)
        conf  = float(top_items[0][1])
        to_gui.put({"type":"voice_capture_step","qid":session['qid'],"token":token,"conf":conf})
        session["steps"] += 1

        if session["qtype"] in ("choice","multi"):
            idx = None
            try:
                idx = session["options_norm"].index(token)
            except ValueError:
                for i, on in enumerate(session["options_norm"]):
                    if token and (token in on or on in token):
                        idx = i; break
            if idx is not None:
                key = session["options"][idx]
                session["counts"][key] = session["counts"].get(key, 0) + 1

    def handle_sigint(sig, frame):
        nonlocal stop
        stop = True
        to_gui.put({"type":"log", "text":"SIGINT received — shutting down…"})
    signal.signal(signal.SIGINT, handle_sigint)

    while not stop:

        try:
            msg_back = from_gui.get_nowait()
            mtyp = msg_back.get("type")
            if mtyp == "quit":
                stop = True; break
            elif mtyp == "config_update":
                if "use_uv_gate" in msg_back:
                    use_uv = bool(msg_back["use_uv_gate"]); gate_cfg.use_uv_gate = use_uv; last_gate_walltime = time.time()
                if "threshold_uV" in msg_back:
                    try: gate_cfg.threshold_uV = float(msg_back["threshold_uV"])
                    except Exception: pass
                if "zscore_thresh" in msg_back:
                    try: gate_cfg.zscore_thresh = float(msg_back["zscore_thresh"])
                    except Exception: pass
                if "min_active_channels" in msg_back:
                    try: gate_cfg.min_active_channels = int(msg_back["min_active_channels"])
                    except Exception: pass
                if "consec_samp" in msg_back:
                    try: gate_cfg.consec_samp = int(msg_back["consec_samp"])
                    except Exception: pass
                if "cooldown_samp" in msg_back:
                    try:
                        cooldown_samp = max(0, int(msg_back["cooldown_samp"]))
                        samples_since_last_trigger = cooldown_samp + 1
                    except Exception: pass
                to_gui.put({"type":"log","text":f"[Config] mode={'µV' if gate_cfg.use_uv_gate else 'z'} | µVthr={gate_cfg.threshold_uV:.1f} | zthr={gate_cfg.zscore_thresh:.2f} | min_ch={gate_cfg.min_active_channels} | consec={gate_cfg.consec_samp} | cooldown={cooldown_samp}"})
            elif mtyp == "switch_board":
                req = str(msg_back.get("board", "")).upper().strip()
                if req in ("CYTON", "SYNTHETIC") and req != current_board_name:
                    to_gui.put({"type": "log", "text": f"[Board] Switching → {req}…"})
                    try: live.close()
                    except Exception: pass
                    live = LiveBoard(serial_port=CONFIG.SERIAL, board_id=req)
                    current_board_name = req
                    fs_used = float(CONFIG.FS_OVERRIDE) if CONFIG.FS_OVERRIDE is not None else float(live.fs)

                    ring_len = max((pre_samp + post_samp) * CONFIG.RING_FACTOR, pre_samp + post_samp + 1000)
                    ring = np.zeros((live.n_ch, ring_len), dtype=np.float32)
                    wptr = 0; filled = 0

                    sel_gate = [c-1 for c in CONFIG.GATE_CHANNELS]
                    sel_gate = [s for s in sel_gate if 0 <= s < live.n_ch]
                    if not sel_gate: sel_gate = list(range(min(4, live.n_ch)))

                    if current_board_name == "SYNTHETIC":
                        gate_cfg.use_uv_gate = False
                        gate_cfg.zscore_thresh = CONFIG.SYNTHETIC_Z_THRESH
                        gate_cfg.min_active_channels = CONFIG.SYNTHETIC_MIN_ACTIVE
                        gate_cfg.consec_samp = CONFIG.SYNTHETIC_CONSEC
                    else:
                        gate_cfg.use_uv_gate = bool(CONFIG.USE_UV_GATE)
                        gate_cfg.zscore_thresh = CONFIG.ZSCORE_THRESH
                        gate_cfg.min_active_channels = CONFIG.MIN_ACTIVE_CHANNELS
                        gate_cfg.consec_samp = CONFIG.GATE_CONSEC_SAMP

                    samples_since_last_trigger = cooldown_samp + 1
                    last_gate_walltime = time.time()

                    to_gui.put({"type": "log", "text": f"[Board] Now connected: {current_board_name} @ {live.fs} Hz"})
                    to_gui.put({"type": "log", "text": f"Streaming channels (row idx): {live.ch_idx}"})
                    to_gui.put({"type": "log", "text": f"Gate channels (0-based rows): {sel_gate}"})
                    to_gui.put({"type": "gate_info",
                                "use_uv": gate_cfg.use_uv_gate,
                                "threshold_uV": gate_cfg.threshold_uV,
                                "zscore_thresh": gate_cfg.zscore_thresh,
                                "min_active_channels": gate_cfg.min_active_channels,
                                "consec_samp": gate_cfg.consec_samp,
                                "cooldown_samp": cooldown_samp,
                                "board": current_board_name})
                    to_gui.put({"type": "init",
                                "class_names": clf.class_names,
                                "info": f"Classes={len(clf.class_names)} | Pre={pre_samp} Post={post_samp} (total={pre_samp+post_samp})",
                                "fs": fs_used,
                                "live_sec": CONFIG.LIVE_PLOT_SEC,
                                "n_signal_ch": len(sel_gate)})
            elif mtyp == "start_voice_capture":
                voice_session["active"] = False
                start_voice_session(msg_back)
            elif mtyp == "blink_confirm":
                if pending_confirmation["active"]:
                    pc = pending_confirmation
                    pending_confirmation["active"] = False
                    to_gui.put({"type": "log", "text": f"[Confirm] Blink received. Accepting '{pc['decision']}' (conf≈{pc['confidence']:.2f})."})
                    to_gui.put({"type": "confirm_prompt_clear"})
                    send_voice_decision(pc["qid"], pc["qtype"], pc["decision"], pc["confidence"])
                else:
                    to_gui.put({"type": "log", "text": "[Confirm] Blink received but no pending decision."})
        except queue.Empty:
            pass


        if pending_confirmation["active"]:
            now = time.time()
            if now - pending_confirmation["start_ts"] > 2.0:
                pc = pending_confirmation
                pending_confirmation["active"] = False
                to_gui.put({"type": "log", "text": f"[Confirm] No blink after 2s for '{pc['decision']}' (conf≈{pc['confidence']:.2f}). Using 'hmmm' instead."})
                to_gui.put({"type": "confirm_prompt_clear"})
                send_voice_decision(pc["qid"], pc["qtype"], "hmmm", 0.0)


        time.sleep(0.06)
        chunk = live.read_chunk()
        N = chunk.shape[1]
        if N == 0:
            empty_reads += 1
            if empty_reads % 20 == 0:
                to_gui.put({"type":"log", "text":"[Stream] Still waiting for samples..."})
            continue
        else:
            empty_reads = 0

        samples_since_last_trigger += N


        if N >= ring_len:
            ring = chunk[:, -ring_len:]; wptr = 0; filled = ring_len
        else:
            end = wptr + N
            if end <= ring_len:
                ring[:, wptr:end] = chunk
            else:
                part = ring_len - wptr
                ring[:, wptr:] = chunk[:, :part]
                ring[:, :N-part] = chunk[:, part:]
            wptr = (wptr + N) % ring_len
            filled = min(ring_len, filled + N)


        to_gui.put({"type": "signal", "chunk": (chunk[np.array(sel_gate), :].astype(np.float32)).tolist()})


        if voice_session["active"]:
            now = time.time()
            if now >= voice_session["next_step"] and have_tail_window(clf.target_len):
                voice_step_infer(voice_session)
                voice_session["next_step"] = now + CONFIG.VOICE_STEP_SEC
            if now >= voice_session["end_time"]:
                finish_voice_session(voice_session)


        if CONFIG.NO_GATE_FALLBACK_SEC > 0 and use_uv and (time.time() - last_gate_walltime > CONFIG.NO_GATE_FALLBACK_SEC):
            use_uv = False; gate_cfg.use_uv_gate = False
            msg = f"No gate in {CONFIG.NO_GATE_FALLBACK_SEC:.0f}s → switching to z-score gating."
            to_gui.put({"type": "log", "text": msg})
            print(f"[Gate] {msg}", flush=True)


        if filled < pre_samp: continue


        if samples_since_last_trigger < cooldown_samp: continue


        view = chunk[np.array(sel_gate), :]
        gate_mat = view if gate_cfg.use_uv_gate else zscore(view, axis=1)
        gate_indices = detect_gate_indices(gate_mat, gate_cfg)
        if not gate_indices:
            continue

        last_gate_walltime = time.time()
        s_in_chunk = gate_indices[0]
        trig_wptr = (wptr - N + s_in_chunk) % ring_len


        dist = (trig_wptr - last_trigger_wptr) % ring_len
        if filled == ring_len and dist < gate_cfg.refractory_samp:
            continue

        trigger_id += 1
        ts_trig = time.strftime("%H:%M:%S")
        print(f"[Trigger {trigger_id}] {ts_trig} Gate detected — capturing window.", flush=True)

        samples_since_last_trigger = 0
        last_trigger_wptr = trig_wptr


        if filled == ring_len:
            post_have = (wptr - trig_wptr) % ring_len
        else:
            post_have = (wptr - trig_wptr) if wptr >= trig_wptr else ring_len - trig_wptr + wptr

        t_wait0 = time.time()
        while post_have < gate_cfg.post_samp and not stop:
            time.sleep(0.02)
            chunk2 = live.read_chunk(); N2 = chunk2.shape[1]
            if N2 > 0:
                samples_since_last_trigger += N2
                if N2 >= ring_len:
                    ring = chunk2[:, -ring_len:]; wptr = 0; filled = ring_len
                else:
                    end = wptr + N2
                    if end <= ring_len:
                        ring[:, wptr:end] = chunk2
                    else:
                        part = ring_len - wptr
                        ring[:, wptr:] = chunk2[:, :part]
                        ring[:, :N2-part] = chunk2[:, part:]
                    wptr = (wptr + N2) % ring_len
                    filled = min(ring_len, filled + N2)
                to_gui.put({"type": "signal", "chunk": (chunk2[np.array(sel_gate), :].astype(np.float32)).tolist()})
                if filled == ring_len:
                    post_have = (wptr - trig_wptr) % ring_len
                else:
                    post_have = (wptr - trig_wptr) if wptr >= trig_wptr else ring_len - trig_wptr + wptr
            if (time.time() - t_wait0) > 0.5:
                to_gui.put({"type":"log", "text": f"Waiting post samples… {post_have}/{gate_cfg.post_samp}"})
                t_wait0 = time.time()


        start = (trig_wptr - gate_cfg.pre_samp) % ring_len
        end   = (trig_wptr + gate_cfg.post_samp) % ring_len
        if (gate_cfg.pre_samp + gate_cfg.post_samp) > filled:
            to_gui.put({"type":"log", "text":"[Warn] Not enough data in ring; skipping this trigger."})
            print(f"[Trigger {trigger_id}] Skipped — not enough data in ring.", flush=True)
            continue

        window = ring[:, start:end] if start < end else np.concatenate([ring[:, start:], ring[:, :end]], axis=1)
        T_expected = gate_cfg.pre_samp + gate_cfg.post_samp
        if window.shape[1] != T_expected:
            if window.shape[1] > T_expected: window = window[:, -T_expected:]
            else: window = np.pad(window, ((0,0),(T_expected - window.shape[1], 0)), mode="constant")


        X = window.astype(np.float32)
        if CONFIG.HP_CUTOFF > 0: X = butter_highpass(X, fs=fs_used, cutoff=float(CONFIG.HP_CUTOFF), order=4)
        if CONFIG.NOTCH_HZ in (50, 60): X = notch(X, fs=fs_used, freq=float(CONFIG.NOTCH_HZ), Q=30.0)
        X = zscore(X, axis=1)
        T_target = clf.target_len
        if T_target is not None and X.shape[1] != T_target:
            X = resample(X, T_target, axis=1)

        try:
            p, infer_dt = clf.predict_proba(X)
        except Exception as e:
            to_gui.put({"type":"log", "text": f"[ERROR] Inference failed: {e}"})
            print(f"[Trigger {trigger_id}] Inference ERROR: {e}", flush=True)
            continue

        top_items = clf.topk(p, k=max(1, CONFIG.TOPK))
        ts = time.strftime("%H:%M:%S")
        info_line = (f"pre={gate_cfg.pre_samp} post={gate_cfg.post_samp} total={T_expected} | "
                     f"gate={'µV' if gate_cfg.use_uv_gate else 'z-score'} | cooldown={cooldown_samp} | "
                     f"infer={infer_dt*1000:.0f} ms")
        to_gui.put({"type": "update",
                    "class_names": clf.class_names,
                    "probs": p.tolist(),
                    "top1": top_items[0],
                    "timestamp": ts,
                    "info": info_line,
                    "logline": f"[{ts}] top1={top_items[0][0].upper()} ({top_items[0][1]*100:.1f}%) | infer={infer_dt*1000:.0f} ms"})

        print(f"[Inference {trigger_id}] {ts}  top1={top_items[0][0].upper()} ({top_items[0][1]*100:.1f}%) | infer={infer_dt*1000:.0f} ms", flush=True)

        try: last_win_model = window[np.array(sel_model), :].astype(np.float32)
        except Exception: last_win_model = window[np.array(sel_gate), :].astype(np.float32)

        to_gui.put({"type": "last_window",
                    "chunk": last_win_model.tolist(),
                    "fs": fs_used,
                    "trigger_idx": gate_cfg.pre_samp,
                    "labels": model_labels})


    try: live.close()
    except Exception: pass
    try: to_gui.put({"type":"stop"})
    except Exception: pass
    try: gp.join(timeout=2.0)
    except Exception: pass


if __name__ == "__main__":
    main()
