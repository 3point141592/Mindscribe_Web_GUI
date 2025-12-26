#!/usr/bin/env python3
"""
Dataset-only evaluator for MindScribe ATCNet checkpoints.

- Loads .npy/.npz samples shaped (C,T) or (T,C), and also supports batched (N,C,T)/(N,T,C).
- Runs model inference; measures latency (model only by default; or end-to-end if enabled).
- Prints mean / median / mode / IQR latency (mode via histogram binning),
  plus 5th and 95th percentile latencies (bottom/top 5%),
  and counts predictions whose top-1 confidence is below a threshold (default 70%).
- Saves visuals: latency histogram (with mean/median/mode), CDF, class counts,
  and per-class median latency bar chart.

Dependencies: numpy, scipy, torch, braindecode, matplotlib, (optional) PyQt5
"""

import os, time, csv, sys
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable
from pathlib import Path
import numpy as np


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt




try:
    import torch
    import torch.nn.functional as F
    import torch.backends.mkldnn as _mkldnn
    _mkldnn.enabled = False
    torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "1")))
except Exception as e:
    raise SystemExit("PyTorch is required. pip install torch") from e

try:
    from braindecode.models import ATCNet
except Exception as e:
    raise SystemExit("braindecode is required. pip install braindecode") from e

from scipy.signal import butter, filtfilt, iirnotch, resample
from scipy.ndimage import uniform_filter1d




class CONFIG:

    CKPT_PATH = r"C:\Users\krivi\OneDrive\Desktop\Mindscribe_fromscratch\main_September_26th\final_model\default_3channel\best_model.pt"


    MODEL_CHANNELS = [1, 2, 3, 4, 5, 6, 7, 8]


    HP_CUTOFF  = 15.0
    NOTCH_HZ   = 0.0


class EVAL:

    DATASET_PATH = r"C:\Users\krivi\OneDrive\Desktop\Mindscribe_fromscratch\main_September_26th\final_model\multi_subject_5words.npz"
    RECURSIVE    = True


    NPZ_KEY      = None
    TRANSPOSE    = False


    DATASET_SFREQ = None
    LIMIT         = 0
    INCLUDE_PREPROC_IN_LATENCY = False


    SAVE_CSV    = r""
    OUTPUT_DIR  = r"./eval_out"
    MODE_BIN_MS = 1.0
    SHOW_FIGS   = False


    ENABLE_QT_STRESS = True


    LOW_CONF_THRESHOLD = 0.70





def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _mode_hist(values_ms: List[float], bin_ms: float) -> float:
    """Histogram-based mode (bin center of the most-populated bin)."""
    a = np.asarray(values_ms, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    w = max(1e-6, float(bin_ms))
    lo, hi = a.min(), a.max()
    edges = np.arange(lo, hi + w*1.000001, w)
    if edges.size < 2:
        return float(a.mean())
    hist, edges = np.histogram(a, bins=edges)
    k = int(hist.argmax())
    return float(0.5*(edges[k] + edges[k+1]))





def stack_features(arr: np.ndarray, sfreq: float, env_ms=20, rms_ms=20) -> np.ndarray:
    """Augment channels with rectified envelope and RMS ⇒ (3C, T)."""
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
    raise TypeError(f"ATCNet instantiation failed; last error: {last_err}")


def load_ckpt(path, n_chans_model, n_classes, n_times, sfreq, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = build_atcnet(n_chans_model, n_classes, n_times, sfreq).to(device)
    state = ckpt.get("state_dict", ckpt)
    clean = {}
    for k, v in state.items():
        if k == "n_averaged" or k.endswith(".n_averaged"):
            continue
        clean[k.replace("module.", "")] = v
    model.load_state_dict(clean, strict=False)
    model.eval()
    meta = {
        "class_names": ckpt.get("class_names", None),
        "sfreq": ckpt.get("sfreq", None),
        "target_len": ckpt.get("target_len", None)
    }
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
        self.sfreq       = float(meta_sfreq) if meta_sfreq is not None else (
            float(sfreq_hint) if sfreq_hint is not None else 250.0
        )
        self.target_len  = int(meta_tlen) if meta_tlen is not None else (
            int(target_len_hint) if target_len_hint is not None else int(2.0 * self.sfreq)
        )

        self.sel_channels = sorted([int(c) - 1 for c in top_channels_1based])
        if not self.sel_channels:
            raise ValueError("MODEL_CHANNELS must contain at least one 1-based channel index.")

        n_classes_guess = max(2, len(self.class_names) or 5)
        self.n_chans_model = 3 * len(self.sel_channels)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, meta = load_ckpt(
            ckpt_path, self.n_chans_model, n_classes_guess,
            self.target_len, self.sfreq, device=str(self.device)
        )


        if meta.get("class_names") is not None:
            self.class_names = [str(s).lower() for s in meta["class_names"]]
        if meta.get("sfreq") is not None:
            self.sfreq = float(meta["sfreq"])
        if meta.get("target_len") is not None:
            self.target_len = int(meta["target_len"])

        self.model = model
        if not self.class_names:

            out_features = getattr(self.model, 'final_layer', None).out_features if hasattr(self.model, 'final_layer') else n_classes_guess
            self.class_names = [f"class_{i}" for i in range(out_features)]

    def _prepare(self, emg_CxT: np.ndarray) -> torch.Tensor:
        a = np.asarray(emg_CxT, dtype=np.float32)
        assert a.ndim == 2, f"Expected (C, T), got {a.shape}"
        a = a[self.sel_channels]
        a = stack_features(a, self.sfreq)
        assert a.shape[0] == self.n_chans_model
        return torch.from_numpy(a).unsqueeze(0).to(self.device)

    def predict_proba(self, emg_CxT: np.ndarray) -> Tuple[np.ndarray, float]:
        """Returns (probs, latency_seconds) — latency is model-forward only."""
        x = self._prepare(emg_CxT)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = self.model(x).view(x.size(0), -1)
            p = F.softmax(out, dim=1).cpu().numpy()[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        return p, dt

    def top1(self, p: np.ndarray) -> Tuple[str, float]:
        i = int(np.argmax(p))
        name = self.class_names[i] if i < len(self.class_names) else f"class_{i}"
        return name, float(p[i])





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


def preprocess_for_model(x_CxT: np.ndarray, fs_in: Optional[float], clf: EMGClassifier) -> np.ndarray:
    X = x_CxT.astype(np.float32, copy=True)
    fs = float(fs_in) if fs_in is not None else float(getattr(clf, "sfreq", 250.0))
    if CONFIG.HP_CUTOFF > 0:
        X = butter_highpass(X, fs=fs, cutoff=float(CONFIG.HP_CUTOFF), order=4)
    if CONFIG.NOTCH_HZ in (50, 60):
        X = notch(X, fs=fs, freq=float(CONFIG.NOTCH_HZ), Q=30.0)
    X = zscore(X, axis=1)
    T_target = int(getattr(clf, "target_len", X.shape[1]))
    if X.shape[1] != T_target:
        X = resample(X, T_target, axis=1)
    return X





FALLBACK_NPZ_KEYS = ("x", "X", "data", "emg", "signals")


def _ensure_CxT(arr: np.ndarray, transpose_flag: bool) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D array per sample, got {a.shape}")
    if transpose_flag:
        a = a.T
    return a.astype(np.float32, copy=False)


def _iter_samples_from_file(path: Path, npz_key: Optional[str], transpose_flag: bool) -> Iterable[Tuple[np.ndarray, str]]:
    """Yield (sample_CxT, sample_id) from .npy/.npz (supports 2D or 3D batch)."""
    if path.suffix.lower() == ".npy":
        arr = np.load(str(path), allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        z = np.load(str(path), allow_pickle=False)
        key = npz_key
        if key is None:
            for k in FALLBACK_NPZ_KEYS:
                if k in z.files:
                    key = k
                    break
            if key is None:
                key = z.files[0]
        arr = z[key]
    else:
        return

    if arr.ndim == 2:
        yield _ensure_CxT(arr, transpose_flag or False), f"{path.name}"
    elif arr.ndim == 3:

        N, A, B = arr.shape

        is_ntc = (A > B)
        for i in range(N):
            s = arr[i]
            if is_ntc:
                s = s.T
            yield _ensure_CxT(s, transpose_flag or False), f"{path.name}#{i}"
    else:
        raise ValueError(f"{path}: unsupported array shape {arr.shape}")


def iter_dataset(dataset_path: Path, recursive: bool, npz_key: Optional[str], transpose_flag: bool) -> Iterable[Tuple[np.ndarray, str]]:
    p = Path(dataset_path)
    if p.is_file():
        yield from _iter_samples_from_file(p, npz_key, transpose_flag)
        return
    if not p.is_dir():
        raise FileNotFoundError(f"Dataset path not found: {p}")
    it = p.rglob("*") if recursive else p.iterdir()
    for f in sorted(it):
        if f.is_file() and f.suffix.lower() in (".npy", ".npz"):
            yield from _iter_samples_from_file(f, npz_key, transpose_flag)





def evaluate_and_save():

    if not CONFIG.CKPT_PATH or not os.path.exists(CONFIG.CKPT_PATH):
        raise FileNotFoundError(f"Edit CONFIG.CKPT_PATH; not found: {CONFIG.CKPT_PATH}")
    if not EVAL.DATASET_PATH:
        raise ValueError("Set EVAL.DATASET_PATH to a folder or a .npy/.npz file.")

    out_dir = Path(EVAL.OUTPUT_DIR or "./eval_out")
    _ensure_dir(out_dir)


    clf = EMGClassifier(
        ckpt_path=CONFIG.CKPT_PATH,
        top_channels_1based=CONFIG.MODEL_CHANNELS,
        class_names_hint=None,
        sfreq_hint=(float(EVAL.DATASET_SFREQ) if EVAL.DATASET_SFREQ is not None else None),
        target_len_hint=None
    )


    try:
        _zeros = np.zeros((len(CONFIG.MODEL_CHANNELS), max(8, clf.target_len)), dtype=np.float32)
        clf.predict_proba(_zeros)
    except Exception:
        pass


    lat_ms: List[float] = []
    rows: List[dict] = []
    per_class_lat = {lbl: [] for lbl in clf.class_names}
    n_total = 0
    low_conf_count = 0

    for x, sid in iter_dataset(Path(EVAL.DATASET_PATH), EVAL.RECURSIVE, EVAL.NPZ_KEY, EVAL.TRANSPOSE):
        try:
            fs_used = float(EVAL.DATASET_SFREQ) if EVAL.DATASET_SFREQ is not None else float(clf.sfreq)

            if EVAL.INCLUDE_PREPROC_IN_LATENCY:
                t0 = time.perf_counter()
                X = preprocess_for_model(x, fs_used, clf)
                p, _dt = clf.predict_proba(X)
                dt_s = time.perf_counter() - t0
            else:
                X = preprocess_for_model(x, fs_used, clf)
                p, dt_s = clf.predict_proba(X)

            top1, top1p = clf.top1(p)
            lat = float(dt_s * 1000.0)


            if float(top1p) < float(EVAL.LOW_CONF_THRESHOLD):
                low_conf_count += 1

            lat_ms.append(lat)
            rows.append({"sample_id": sid, "top1": top1, "top1_prob": float(top1p), "latency_ms": lat})
            if top1 in per_class_lat:
                per_class_lat[top1].append(lat)
            else:
                per_class_lat[top1] = [lat]

            n_total += 1
            if EVAL.LIMIT and n_total >= int(EVAL.LIMIT):
                break
        except Exception as e:
            rows.append({"sample_id": sid, "error": str(e)})


    good = [r["latency_ms"] for r in rows if "latency_ms" in r]
    if good:
        a = np.asarray(good, dtype=np.float64)
        mean_ms   = float(a.mean())
        median_ms = float(np.median(a))
        mode_ms   = _mode_hist(good, EVAL.MODE_BIN_MS)


        q1_ms = float(np.percentile(a, 25))
        q3_ms = float(np.percentile(a, 75))
        iqr_ms = q3_ms - q1_ms


        p5_ms  = float(np.percentile(a, 5))
        p95_ms = float(np.percentile(a, 95))
    else:
        mean_ms = median_ms = mode_ms = float("nan")
        q1_ms = q3_ms = iqr_ms = float("nan")
        p5_ms = p95_ms = float("nan")


    n_preds = sum(1 for r in rows if "top1_prob" in r)
    low_conf_pct = (100.0 * low_conf_count / n_preds) if n_preds else 0.0


    print("\n=== Latency statistics (milliseconds) ===")
    print(f"Samples evaluated: {len(good)} / total rows: {len(rows)}")
    print(f"Mean        : {mean_ms:.3f}")
    print(f"Median      : {median_ms:.3f}")
    print(f"Mode        : {mode_ms:.3f}  (bin={EVAL.MODE_BIN_MS:.3f} ms)")
    print(f"Q1 (25th)   : {q1_ms:.3f}")
    print(f"Q3 (75th)   : {q3_ms:.3f}")
    print(f"IQR         : {iqr_ms:.3f}")
    print(f"Bottom 5%   : {p5_ms:.3f}  (5th percentile)")
    print(f"Top 5%      : {p95_ms:.3f} (95th percentile)")
    print(f"Low-confidence top-1 (<{EVAL.LOW_CONF_THRESHOLD*100:.0f}%): "
          f"{low_conf_count} of {n_preds} ({low_conf_pct:.1f}%)")


    if EVAL.SAVE_CSV:
        csv_path = Path(EVAL.SAVE_CSV)
        _ensure_dir(csv_path.parent)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["sample_id", "top1", "top1_prob", "latency_ms", "error"])
            w.writeheader()
            for r in rows:
                w.writerow({
                    "sample_id": r.get("sample_id", ""),
                    "top1": r.get("top1", ""),
                    "top1_prob": r.get("top1_prob", ""),
                    "latency_ms": r.get("latency_ms", ""),
                    "error": r.get("error", ""),
                })
        print(f"Per-sample results saved to: {csv_path.resolve()}")



    if good:
        fig = plt.figure(figsize=(8, 5))
        plt.hist(good, bins=50)
        plt.axvline(mean_ms,   linestyle="--", linewidth=1.2, label=f"Mean {mean_ms:.1f} ms")
        plt.axvline(median_ms, linestyle="--", linewidth=1.2, label=f"Median {median_ms:.1f} ms")
        plt.axvline(mode_ms,   linestyle="--", linewidth=1.2, label=f"Mode {mode_ms:.1f} ms")
        plt.title("Latency Histogram (ms)")
        plt.xlabel("Latency (ms)")
        plt.ylabel("Count")
        plt.legend()
        p = out_dir / "latency_hist.png"
        plt.tight_layout()
        plt.savefig(p, dpi=150)
        if EVAL.SHOW_FIGS:
            plt.show()
        plt.close(fig)


        fig = plt.figure(figsize=(8, 5))
        xs = np.sort(np.asarray(good, dtype=np.float64))
        ys = np.linspace(0, 1, num=len(xs), endpoint=True)
        plt.plot(xs, ys, linewidth=1.5)
        plt.title("Latency CDF")
        plt.xlabel("Latency (ms)")
        plt.ylabel("Cumulative probability")
        p = out_dir / "latency_cdf.png"
        plt.tight_layout()
        plt.savefig(p, dpi=150)
        if EVAL.SHOW_FIGS:
            plt.show()
        plt.close(fig)


        counts = {}
        for r in rows:
            if "top1" in r:
                counts[r["top1"]] = counts.get(r["top1"], 0) + 1
        if counts:
            labels = list(sorted(counts.keys()))
            vals = [counts[k] for k in labels]
            fig = plt.figure(figsize=(9, 5))
            x = np.arange(len(labels))
            plt.bar(x, vals)
            plt.xticks(x, [str(s).upper() for s in labels], rotation=45, ha="right")
            plt.title("Predicted Class Counts")
            plt.xlabel("Class")
            plt.ylabel("Count")
            p = out_dir / "class_counts.png"
            plt.tight_layout()
            plt.savefig(p, dpi=150)
            if EVAL.SHOW_FIGS:
                plt.show()
            plt.close(fig)


        med_per_class = {k: (float(np.median(v)) if v else np.nan) for k, v in per_class_lat.items() if v}
        if med_per_class:
            labels = list(sorted(med_per_class.keys()))
            vals = [med_per_class[k] for k in labels]
            fig = plt.figure(figsize=(9, 5))
            x = np.arange(len(labels))
            plt.bar(x, vals)
            plt.xticks(x, [str(s).upper() for s in labels], rotation=45, ha="right")
            plt.title("Per-class Median Latency (ms)")
            plt.xlabel("Class")
            plt.ylabel("Median latency (ms)")
            p = out_dir / "per_class_median_latency.png"
            plt.tight_layout()
            plt.savefig(p, dpi=150)
            if EVAL.SHOW_FIGS:
                plt.show()
            plt.close(fig)


    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("Latency statistics (milliseconds)\n")
        f.write(f"Samples: {len(good)} / total rows: {len(rows)}\n")
        f.write(f"Mean        : {mean_ms:.3f}\n")
        f.write(f"Median      : {median_ms:.3f}\n")
        f.write(f"Mode        : {mode_ms:.3f} (bin={EVAL.MODE_BIN_MS:.3f} ms)\n")
        f.write(f"Q1 (25th)   : {q1_ms:.3f}\n")
        f.write(f"Q3 (75th)   : {q3_ms:.3f}\n")
        f.write(f"IQR         : {iqr_ms:.3f}\n")
        f.write(f"Bottom 5%   : {p5_ms:.3f} (5th percentile)\n")
        f.write(f"Top 5%      : {p95_ms:.3f} (95th percentile)\n")
        f.write(f"Low-confidence top-1 (<{EVAL.LOW_CONF_THRESHOLD*100:.0f}%): "
                f"{low_conf_count} of {n_preds} ({low_conf_pct:.1f}%)\n")

    print(f"Figures & summary saved to: {out_dir.resolve()}")





if __name__ == "__main__":
    if getattr(EVAL, "ENABLE_QT_STRESS", False):
        try:
            from PyQt5.QtWidgets import QApplication, QLabel
            from PyQt5.QtCore import QTimer
        except Exception as e:
            print(f"PyQt5 not available ({e}); running without Qt stress window.")
            evaluate_and_save()
        else:
            import threading

            app = QApplication(sys.argv)

            label = QLabel(
                "MindScribe latency evaluator is running.\n"
                "This window is here to simulate GUI load (PyQt) during inference."
            )
            label.setWindowTitle("MindScribe Stress Window")
            label.resize(420, 180)
            label.show()

            def run_eval():
                try:
                    evaluate_and_save()
                finally:

                    QTimer.singleShot(0, app.quit)

            t = threading.Thread(target=run_eval, daemon=True)
            t.start()


            sys.exit(app.exec_())
    else:
        evaluate_and_save()
