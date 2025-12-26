#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mindscribe_core.py — MindScribe core loop (BrainFlow/Cyton or Synthetic) + EMG classifier.

This is your original script with the Tkinter GUI removed and replaced by the 'bridge' message bus.
The web front end (templates/index.html) connects over WebSocket and sends/receives the same message
types that your Tk GUI used internally.

Key differences vs the original:
  - No multiprocessing GUI process.
  - No OpenCV/MediaPipe on the backend (camera/blink runs in browser).
  - No Kokoro/pyttsx3 backend TTS (browser Web Speech API handles TTS).
  - The core thread does NOT install SIGINT handlers when running under Flask
    (signals only work in the main thread). Use the web "Quit" button to stop.

You still keep:
  - Live board streaming (BrainFlow when available, fallback synthetic source)
  - Z-score / µV gating
  - Ring buffer window capture (pre/post)
  - ATCNet inference via braindecode
  - Conversation flow + blink confirmation logic (blink_confirm command comes from browser)

Message types emitted to the UI:
  init, gate_info, signal, update, last_window, convo_spec, voice_capture_step,
  voice_decision, confirm_prompt, confirm_prompt_clear, log

Commands accepted from UI:
  gui_ready, quit, config_update, switch_board, start_voice_capture, blink_confirm
"""

import os, time, queue, copy, threading
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

# Thread limits (optional)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# -------- bridge (web UI transport) --------
import bridge


# -------- ML / DSP deps --------
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

# -------- BrainFlow optional --------
try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    _HAS_BRAINFLOW = True
except Exception:
    BoardShim = BrainFlowInputParams = BoardIds = None
    _HAS_BRAINFLOW = False


class CONFIG:
    # EDIT THESE:
    CKPT_PATH = r"final_model\default_3channel\best_model.pt"
    SERIAL    = "COM6"
    BOARD_MODE = "SYNTHETIC"  # "CYTON" or "SYNTHETIC" (or "CYTON_DAISY")

    USED_CHANNELS  = [1, 2, 3, 4, 5, 6, 7, 8]
    MODEL_CHANNELS = [1, 2, 3, 4, 5, 6, 7, 8]
    GATE_CHANNELS  = [1, 2, 3, 4]

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

    SYNTHETIC_Z_THRESH   = 1.5
    SYNTHETIC_MIN_ACTIVE = 1
    SYNTHETIC_CONSEC     = 3

    NO_GATE_FALLBACK_SEC = 15.0

    TOPK = 5
    FS_OVERRIDE = None

    RING_FACTOR = 3
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


# ---------------- Board source ----------------

class LiveBoard:
    def __init__(self, serial_port: str, board_id: str = "SYNTHETIC"):
        name = str(board_id).upper()
        self.mode = name
        if _HAS_BRAINFLOW:
            BoardShim.enable_dev_board_logger()
            params = BrainFlowInputParams()
            params.serial_port = serial_port or ""
            if name == "CYTON_DAISY":
                bid = BoardIds.CYTON_DAISY_BOARD.value
            elif name == "CYTON":
                bid = BoardIds.CYTON_BOARD.value
            else:
                bid = BoardIds.SYNTHETIC_BOARD.value

            self.board = BoardShim(bid, params)
            self.board.prepare_session()
            self.board.start_stream()
            time.sleep(0.3)

            self.fs = float(BoardShim.get_sampling_rate(bid))
            chs = BoardShim.get_eeg_channels(bid)
            if not chs:
                try:
                    chs = BoardShim.get_exg_channels(bid)
                except Exception:
                    chs = []
            self.ch_idx = chs
            self.n_ch = len(chs)
            if self.n_ch == 0:
                raise RuntimeError(f"No EEG/EXG channels for board {name} (id={bid}).")
            self._chunk = max(1, int(self.fs * 0.12))
            print(f"[Live] Connected: board={name}, fs={self.fs:.1f} Hz, channels={self.ch_idx}")
        else:
            # built-in synthetic source
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

        # synthetic
        N = self._chunk
        sig = 10.0 * np.random.randn(self.n_ch, N).astype(np.float32)
        self._t += N
        return sig

    def close(self):
        if _HAS_BRAINFLOW and hasattr(self, "board"):
            try:
                self.board.stop_stream()
            except Exception:
                pass
            try:
                self.board.release_session()
            except Exception:
                pass


# ---------------- Feature stack / model ----------------

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
        try:
            return ATCNet(**kw)
        except Exception as e:
            last_err = e
    raise TypeError(f"Could not instantiate ATCNet; last error: {last_err}")


def load_ckpt(path, n_chans_model, n_classes, n_times, sfreq, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = build_atcnet(n_chans_model, n_classes, n_times, sfreq).to(device)
    state = ckpt.get("state_dict", ckpt)
    clean_state = {}
    for k, v in state.items():
        if k == "n_averaged" or k.endswith(".n_averaged"):
            continue
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
            raise ValueError("No MODEL_CHANNELS provided. Set CONFIG.MODEL_CHANNELS.")

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


# ---------------- DSP helpers ----------------

def butter_highpass(sig: np.ndarray, fs: float, cutoff: float = 15.0, order: int = 4) -> np.ndarray:
    if cutoff <= 0:
        return sig
    ny = 0.5 * fs
    Wn = cutoff / ny
    b, a = butter(order, Wn, btype='highpass')
    return filtfilt(b, a, sig, axis=1)

def notch(sig: np.ndarray, fs: float, freq: float = 60.0, Q: float = 30.0) -> np.ndarray:
    if freq <= 0:
        return sig
    b, a = iirnotch(w0=freq/(fs/2.0), Q=Q)
    return filtfilt(b, a, sig, axis=1)

def zscore(sig: np.ndarray, axis=1, eps=1e-6) -> np.ndarray:
    m = sig.mean(axis=axis, keepdims=True)
    s = sig.std(axis=axis, keepdims=True)
    return (sig - m) / (s + eps)


# ---------------- Gate detection ----------------

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
    if N == 0:
        return []
    M = max(1, int(cfg.consec_samp))
    if N < M:
        return []
    thr = cfg.threshold_uV if cfg.use_uv_gate else cfg.zscore_thresh
    above = (np.abs(chunk_vals) >= thr).astype(np.int16)
    kern = np.ones(M, dtype=np.int16)
    runs = np.zeros((C, N - M + 1), dtype=np.int16)
    for c in range(C):
        runs[c] = np.convolve(above[c], kern, mode='valid')
    chan_ok = (runs == M).sum(axis=0)
    idx = np.where(chan_ok >= cfg.min_active_channels)[0]
    if idx.size == 0:
        return []
    return [int(idx[0] + (M - 1))]


# ---------------- Vocabulary restriction helpers ----------------

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
        if default_next is None and f:
            default_next = next(iter(f.values()))
        if default_next is None:
            default_next = None
        f_out = {"_any": default_next}
        for opt in q["options"]:
            f_out[opt] = f.get(opt, default_next)
        q["followups"] = f_out
    return spec


# ---------------- Main core loop ----------------

def run():
    # ---- Wait for UI (optional) ----
    bridge.emit({"type":"log","text":"[Init] MindScribe core starting..."})
    if not bridge.gui_ready.wait(timeout=10.0):
        bridge.emit({"type":"log","text":"[Init] gui_ready not received within 10s; continuing anyway."})

    if not os.path.exists(CONFIG.CKPT_PATH):
        raise FileNotFoundError(f"Edit CONFIG.CKPT_PATH to point to your checkpoint. Not found: {CONFIG.CKPT_PATH}")

    current_board_name = str(CONFIG.BOARD_MODE).upper().strip()
    live = LiveBoard(serial_port=CONFIG.SERIAL, board_id=current_board_name)

    fs_used = float(CONFIG.FS_OVERRIDE) if CONFIG.FS_OVERRIDE is not None else float(live.fs)

    sel_gate = [c-1 for c in CONFIG.GATE_CHANNELS]
    sel_gate = [s for s in sel_gate if 0 <= s < live.n_ch]
    if not sel_gate:
        sel_gate = list(range(min(4, live.n_ch)))

    bridge.emit({"type":"log","text":f"Starting board: {current_board_name} @ {live.fs} Hz"})
    bridge.emit({"type":"log","text":f"Streaming channels (row idx): {live.ch_idx}"})
    bridge.emit({"type":"log","text":f"Gate channels (0-based rows): {sel_gate}"})
    bridge.emit({"type":"log","text":f"USED_CHANNELS={CONFIG.USED_CHANNELS}  MODEL={CONFIG.MODEL_CHANNELS}  GATE={CONFIG.GATE_CHANNELS}"})

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
        bridge.emit({"type":"convo_spec","spec": conv_spec_restricted})

    # warm-up
    try:
        bridge.emit({"type":"log","text":"Warming up model…"})
        _zeros = np.zeros((len(CONFIG.MODEL_CHANNELS), max(8, clf.target_len)), dtype=np.float32)
        _ = clf.predict_proba(_zeros)
        bridge.emit({"type":"log","text":"Model warm-up complete."})
    except Exception as e:
        bridge.emit({"type":"log","text":f"[Warm-up skipped] {e}"})

    # init message to UI
    bridge.emit({
        "type":"init",
        "class_names": clf.class_names,
        "info": f"Classes={len(clf.class_names)} | Pre={CONFIG.PRE_SAMP} Post={CONFIG.POST_SAMP} (total={CONFIG.PRE_SAMP+CONFIG.POST_SAMP})",
        "fs": fs_used,
        "live_sec": CONFIG.LIVE_PLOT_SEC,
        "n_signal_ch": len(sel_gate),
    })

    pre_samp = int(CONFIG.PRE_SAMP)
    post_samp = int(CONFIG.POST_SAMP)
    total = pre_samp + post_samp
    refractory_samp = int(CONFIG.REFRACTORY_SAMP) if CONFIG.REFRACTORY_SAMP is not None else max(1, post_samp // 2)
    cooldown_samp = int(CONFIG.COOLDOWN_SAMP)

    gate_cfg = GateConfig(
        threshold_uV=50.0,
        min_active_channels=CONFIG.MIN_ACTIVE_CHANNELS,
        pre_samp=pre_samp,
        post_samp=post_samp,
        refractory_samp=refractory_samp,
        use_uv_gate=(CONFIG.USE_UV_GATE and (current_board_name != "SYNTHETIC")),
        zscore_thresh=CONFIG.ZSCORE_THRESH,
        consec_samp=CONFIG.GATE_CONSEC_SAMP
    )

    if current_board_name == "SYNTHETIC":
        gate_cfg.use_uv_gate = False
        gate_cfg.zscore_thresh = CONFIG.SYNTHETIC_Z_THRESH
        gate_cfg.min_active_channels = CONFIG.SYNTHETIC_MIN_ACTIVE
        gate_cfg.consec_samp = CONFIG.SYNTHETIC_CONSEC
        bridge.emit({"type":"log","text":f"[Synthetic gate] |z| ≥ {gate_cfg.zscore_thresh}, min_channels={gate_cfg.min_active_channels}, consecutive={gate_cfg.consec_samp}"})
    else:
        bridge.emit({"type":"log","text":f"[Real gate] mode={'µV' if gate_cfg.use_uv_gate else 'z-score'}, thr={gate_cfg.threshold_uV if gate_cfg.use_uv_gate else gate_cfg.zscore_thresh}, min_channels={gate_cfg.min_active_channels}, consecutive={gate_cfg.consec_samp}"})

    bridge.emit({"type":"log","text":f"Window: pre={pre_samp}, post={post_samp}, total={total}. Refractory={refractory_samp}. Cooldown={cooldown_samp}."})

    bridge.emit({
        "type":"gate_info",
        "use_uv": gate_cfg.use_uv_gate,
        "threshold_uV": gate_cfg.threshold_uV,
        "zscore_thresh": gate_cfg.zscore_thresh,
        "min_active_channels": gate_cfg.min_active_channels,
        "consec_samp": gate_cfg.consec_samp,
        "cooldown_samp": cooldown_samp,
        "board": current_board_name,
    })

    ring_len = max((pre_samp + post_samp) * CONFIG.RING_FACTOR, pre_samp + post_samp + 1000)
    ring = np.zeros((live.n_ch, ring_len), dtype=np.float32)
    wptr = 0
    filled = 0

    sel_model = [c-1 for c in CONFIG.MODEL_CHANNELS]
    model_labels = [f"CH{c}" for c in CONFIG.MODEL_CHANNELS]

    last_trigger_wptr = -10_000_000
    samples_since_last_trigger = cooldown_samp + 1
    last_gate_walltime = time.time()
    use_uv = gate_cfg.use_uv_gate
    stop = False
    trigger_id = 0
    empty_reads = 0

    # Voice session state (driven by UI start_voice_capture)
    voice_session = {
        "active": False,
        "qid": None,
        "qtype": None,
        "options": [],
        "options_norm": [],
        "min": 0,
        "max": 10,
        "end_time": 0.0,
        "next_step": 0.0,
        "counts": {},
        "numbers": [],
        "texts": [],
        "steps": 0,
    }

    URGENT_TOKENS = {"pain"}
    CONFIRM_THRESHOLD = 0.70
    pending_confirmation = {"active": False, "qid": None, "qtype": None, "decision": "", "confidence": 0.0, "start_ts": 0.0}

    def have_tail_window(T_needed: int) -> bool:
        return filled >= T_needed

    def get_tail_window(T_needed: int) -> Optional[np.ndarray]:
        nonlocal wptr, filled, ring
        if filled < T_needed:
            return None
        start = (wptr - T_needed) % ring_len
        end_ = wptr % ring_len
        if start < end_:
            return ring[:, start:end_].copy()
        else:
            return np.concatenate([ring[:, start:], ring[:, :end_]], axis=1).copy()

    def send_voice_decision(qid, qtype, decision, confidence):
        bridge.emit({"type":"voice_decision","qid":qid,"qtype":qtype,"decision":decision,"confidence":confidence})
        bridge.emit({"type":"log","text":f"[Voice] Final decision {qid} → {decision} (conf≈{confidence:.2f})"})

    def maybe_require_confirmation(qid, qtype, decision, confidence):
        nonlocal pending_confirmation
        if not decision:
            send_voice_decision(qid, qtype, "hmmm", 0.0)
            return
        token = (decision or "").lower()
        is_urgent = token in URGENT_TOKENS
        if is_urgent or confidence >= CONFIRM_THRESHOLD:
            send_voice_decision(qid, qtype, decision, confidence)
            return
        pending_confirmation = {"active": True, "qid": qid, "qtype": qtype, "decision": decision, "confidence": confidence, "start_ts": time.time()}
        bridge.emit({"type":"confirm_prompt","word":decision,"confidence":confidence})
        bridge.emit({"type":"log","text":f"[Confirm] Underconfident non-urgent '{decision}' (conf≈{confidence:.2f}). Waiting for blink (2s)…"})

    def finish_voice_session(session: dict):
        qid = session["qid"]
        qtype = session["qtype"]
        decision = ""
        confidence = 0.0
        counts = session["counts"]
        if counts:
            sorted_items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            decision, c = sorted_items[0]
            total_ = sum(counts.values())
            confidence = c / total_ if total_ > 0 else 0.0
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
        bridge.emit({"type":"log","text":f"[Voice] Capture start {voice_session['qid']} for {CONFIG.VOICE_CAPTURE_SEC:.1f}s"})

    def voice_step_infer(session: dict):
        T_target = clf.target_len
        X = get_tail_window(T_target)
        if X is None:
            return
        if CONFIG.HP_CUTOFF > 0:
            X = butter_highpass(X, fs=fs_used, cutoff=float(CONFIG.HP_CUTOFF), order=4)
        if CONFIG.NOTCH_HZ in (50, 60):
            X = notch(X, fs=fs_used, freq=float(CONFIG.NOTCH_HZ), Q=30.0)
        X = zscore(X, axis=1)
        if X.shape[1] != T_target:
            X = resample(X, T_target, axis=1)
        try:
            p, _ = clf.predict_proba(X)
        except Exception as e:
            bridge.emit({"type":"log","text":f"[Voice] step infer error: {e}"})
            return

        top_items = clf.topk(p, k=1)
        token_raw = str(top_items[0][0]).strip().lower()
        token = norm_token(token_raw)
        conf  = float(top_items[0][1])

        bridge.emit({"type":"voice_capture_step","qid":session["qid"],"token":token,"conf":conf})
        session["steps"] += 1

        if session["qtype"] in ("choice","multi"):
            idx = None
            try:
                idx = session["options_norm"].index(token)
            except ValueError:
                for i, on in enumerate(session["options_norm"]):
                    if token and (token in on or on in token):
                        idx = i
                        break
            if idx is not None:
                key = session["options"][idx]
                session["counts"][key] = session["counts"].get(key, 0) + 1

    # Run loop
    bridge.emit({"type":"log","text":"[Run] Streaming. Use web UI Quit to stop."})
    print("[Run] Streaming. Use web UI Quit to stop.")

    while not stop:
        # ---- handle commands from web UI ----
        msg_back = bridge.poll_cmd_nowait()
        while msg_back is not None:
            mtyp = msg_back.get("type")

            if mtyp == "quit":
                stop = True
                bridge.emit({"type":"log","text":"Quit received — shutting down…"})
                break

            elif mtyp == "config_update":
                if "use_uv_gate" in msg_back:
                    use_uv = bool(msg_back["use_uv_gate"])
                    gate_cfg.use_uv_gate = use_uv
                    last_gate_walltime = time.time()
                if "threshold_uV" in msg_back:
                    try:
                        gate_cfg.threshold_uV = float(msg_back["threshold_uV"])
                    except Exception:
                        pass
                if "zscore_thresh" in msg_back:
                    try:
                        gate_cfg.zscore_thresh = float(msg_back["zscore_thresh"])
                    except Exception:
                        pass
                if "min_active_channels" in msg_back:
                    try:
                        gate_cfg.min_active_channels = int(msg_back["min_active_channels"])
                    except Exception:
                        pass
                if "consec_samp" in msg_back:
                    try:
                        gate_cfg.consec_samp = int(msg_back["consec_samp"])
                    except Exception:
                        pass
                if "cooldown_samp" in msg_back:
                    try:
                        cooldown_samp = max(0, int(msg_back["cooldown_samp"]))
                        samples_since_last_trigger = cooldown_samp + 1
                    except Exception:
                        pass
                bridge.emit({"type":"log","text":f"[Config] mode={'µV' if gate_cfg.use_uv_gate else 'z'} | µVthr={gate_cfg.threshold_uV:.1f} | zthr={gate_cfg.zscore_thresh:.2f} | min_ch={gate_cfg.min_active_channels} | consec={gate_cfg.consec_samp} | cooldown={cooldown_samp}"})

            elif mtyp == "switch_board":
                req = str(msg_back.get("board", "")).upper().strip()
                if req in ("CYTON", "SYNTHETIC", "CYTON_DAISY") and req != current_board_name:
                    bridge.emit({"type":"log","text":f"[Board] Switching → {req}…"})
                    try:
                        live.close()
                    except Exception:
                        pass
                    live = LiveBoard(serial_port=CONFIG.SERIAL, board_id=req)
                    current_board_name = req
                    fs_used = float(CONFIG.FS_OVERRIDE) if CONFIG.FS_OVERRIDE is not None else float(live.fs)

                    ring_len = max((pre_samp + post_samp) * CONFIG.RING_FACTOR, pre_samp + post_samp + 1000)
                    ring = np.zeros((live.n_ch, ring_len), dtype=np.float32)
                    wptr = 0
                    filled = 0

                    sel_gate = [c-1 for c in CONFIG.GATE_CHANNELS]
                    sel_gate = [s for s in sel_gate if 0 <= s < live.n_ch]
                    if not sel_gate:
                        sel_gate = list(range(min(4, live.n_ch)))

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

                    bridge.emit({"type":"log","text":f"[Board] Now connected: {current_board_name} @ {live.fs} Hz"})
                    bridge.emit({"type":"gate_info",
                                 "use_uv": gate_cfg.use_uv_gate,
                                 "threshold_uV": gate_cfg.threshold_uV,
                                 "zscore_thresh": gate_cfg.zscore_thresh,
                                 "min_active_channels": gate_cfg.min_active_channels,
                                 "consec_samp": gate_cfg.consec_samp,
                                 "cooldown_samp": cooldown_samp,
                                 "board": current_board_name})
                    bridge.emit({"type":"init",
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
                    bridge.emit({"type":"log","text":f"[Confirm] Blink received. Accepting '{pc['decision']}' (conf≈{pc['confidence']:.2f})."})
                    bridge.emit({"type":"confirm_prompt_clear"})
                    send_voice_decision(pc["qid"], pc["qtype"], pc["decision"], pc["confidence"])
                else:
                    bridge.emit({"type":"log","text":"[Confirm] Blink received but no pending decision."})

            msg_back = bridge.poll_cmd_nowait()

        # ---- confirmation timeout ----
        if pending_confirmation["active"]:
            now = time.time()
            if now - pending_confirmation["start_ts"] > 2.0:
                pc = pending_confirmation
                pending_confirmation["active"] = False
                bridge.emit({"type":"log","text":f"[Confirm] No blink after 2s for '{pc['decision']}' (conf≈{pc['confidence']:.2f}). Using 'hmmm' instead."})
                bridge.emit({"type":"confirm_prompt_clear"})
                send_voice_decision(pc["qid"], pc["qtype"], "hmmm", 0.0)

        # ---- streaming tick ----
        time.sleep(0.06)
        chunk = live.read_chunk()
        N = chunk.shape[1]
        if N == 0:
            empty_reads += 1
            if empty_reads % 20 == 0:
                bridge.emit({"type":"log","text":"[Stream] Still waiting for samples..."})
            continue
        else:
            empty_reads = 0

        samples_since_last_trigger += N

        # ring append
        if N >= ring_len:
            ring = chunk[:, -ring_len:]
            wptr = 0
            filled = ring_len
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

        # send live signal for plotting (gate channels only)
        try:
            bridge.emit({"type":"signal", "chunk": (chunk[np.array(sel_gate), :].astype(np.float32)).tolist()})
        except Exception:
            pass

        # voice capture step inference
        if voice_session["active"]:
            now = time.time()
            if now >= voice_session["next_step"] and have_tail_window(clf.target_len):
                voice_step_infer(voice_session)
                voice_session["next_step"] = now + CONFIG.VOICE_STEP_SEC
            if now >= voice_session["end_time"]:
                finish_voice_session(voice_session)

        # fallback: if no gate for a while in µV mode
        if CONFIG.NO_GATE_FALLBACK_SEC > 0 and use_uv and (time.time() - last_gate_walltime > CONFIG.NO_GATE_FALLBACK_SEC):
            use_uv = False
            gate_cfg.use_uv_gate = False
            msg = f"No gate in {CONFIG.NO_GATE_FALLBACK_SEC:.0f}s → switching to z-score gating."
            bridge.emit({"type":"log","text":msg})
            print(f"[Gate] {msg}", flush=True)

        # gate disabled until enough pre samples in ring
        if filled < pre_samp:
            continue

        # cooldown
        if samples_since_last_trigger < cooldown_samp:
            continue

        # gate detection within this chunk
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

        # wait until we have post samples after trigger
        if filled == ring_len:
            post_have = (wptr - trig_wptr) % ring_len
        else:
            post_have = (wptr - trig_wptr) if wptr >= trig_wptr else ring_len - trig_wptr + wptr

        t_wait0 = time.time()
        while post_have < gate_cfg.post_samp and not stop:
            time.sleep(0.02)
            chunk2 = live.read_chunk()
            N2 = chunk2.shape[1]
            if N2 > 0:
                samples_since_last_trigger += N2
                if N2 >= ring_len:
                    ring = chunk2[:, -ring_len:]
                    wptr = 0
                    filled = ring_len
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

                try:
                    bridge.emit({"type":"signal", "chunk": (chunk2[np.array(sel_gate), :].astype(np.float32)).tolist()})
                except Exception:
                    pass

                if filled == ring_len:
                    post_have = (wptr - trig_wptr) % ring_len
                else:
                    post_have = (wptr - trig_wptr) if wptr >= trig_wptr else ring_len - trig_wptr + wptr

            if (time.time() - t_wait0) > 0.5:
                bridge.emit({"type":"log","text":f"Waiting post samples… {post_have}/{gate_cfg.post_samp}"})
                t_wait0 = time.time()

        start = (trig_wptr - gate_cfg.pre_samp) % ring_len
        end   = (trig_wptr + gate_cfg.post_samp) % ring_len
        if (gate_cfg.pre_samp + gate_cfg.post_samp) > filled:
            bridge.emit({"type":"log","text":"[Warn] Not enough data in ring; skipping this trigger."})
            print(f"[Trigger {trigger_id}] Skipped — not enough data in ring.", flush=True)
            continue

        window = ring[:, start:end] if start < end else np.concatenate([ring[:, start:], ring[:, :end]], axis=1)
        T_expected = gate_cfg.pre_samp + gate_cfg.post_samp
        if window.shape[1] != T_expected:
            if window.shape[1] > T_expected:
                window = window[:, -T_expected:]
            else:
                window = np.pad(window, ((0,0),(T_expected - window.shape[1], 0)), mode="constant")

        # preprocessing + inference
        X = window.astype(np.float32)
        if CONFIG.HP_CUTOFF > 0:
            X = butter_highpass(X, fs=fs_used, cutoff=float(CONFIG.HP_CUTOFF), order=4)
        if CONFIG.NOTCH_HZ in (50, 60):
            X = notch(X, fs=fs_used, freq=float(CONFIG.NOTCH_HZ), Q=30.0)
        X = zscore(X, axis=1)
        T_target = clf.target_len
        if T_target is not None and X.shape[1] != T_target:
            X = resample(X, T_target, axis=1)

        try:
            p, infer_dt = clf.predict_proba(X)
        except Exception as e:
            bridge.emit({"type":"log","text":f"[ERROR] Inference failed: {e}"})
            print(f"[Trigger {trigger_id}] Inference ERROR: {e}", flush=True)
            continue

        top_items = clf.topk(p, k=max(1, CONFIG.TOPK))
        ts = time.strftime("%H:%M:%S")
        info_line = (f"pre={gate_cfg.pre_samp} post={gate_cfg.post_samp} total={T_expected} | "
                     f"gate={'µV' if gate_cfg.use_uv_gate else 'z-score'} | cooldown={cooldown_samp} | "
                     f"infer={infer_dt*1000:.0f} ms")

        bridge.emit({
            "type":"update",
            "class_names": clf.class_names,
            "probs": p.tolist(),
            "top1": top_items[0],
            "timestamp": ts,
            "info": info_line,
            "logline": f"[{ts}] top1={top_items[0][0].upper()} ({top_items[0][1]*100:.1f}%) | infer={infer_dt*1000:.0f} ms"
        })

        print(f"[Inference {trigger_id}] {ts}  top1={top_items[0][0].upper()} ({top_items[0][1]*100:.1f}%) | infer={infer_dt*1000:.0f} ms", flush=True)

        try:
            last_win_model = window[np.array(sel_model), :].astype(np.float32)
        except Exception:
            last_win_model = window[np.array(sel_gate), :].astype(np.float32)

        bridge.emit({
            "type":"last_window",
            "chunk": last_win_model.tolist(),
            "fs": fs_used,
            "trigger_idx": gate_cfg.pre_samp,
            "labels": model_labels
        })

    # ---- shutdown ----
    try:
        live.close()
    except Exception:
        pass

    bridge.emit({"type":"log","text":"Core stopped."})
    print("[Run] Core stopped.", flush=True)


if __name__ == "__main__":
    run()
