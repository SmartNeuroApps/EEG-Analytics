"""
run_analysis.py -- EEG attention / cognitive-load analysis (interactive).

Companion code for "A Software Architecture for EEG-Enabled Business
Applications" (EMOTIV EPOC, 14 channels, 128 Hz).

Run:  python run_analysis.py            (answer the menus)
      python run_analysis.py --help     (non-interactive flags)

The anonymized recordings live in ./data next to this script; nothing is
downloaded. The menus ask, in order:
  1. Which data to load: attention, cognitive load, or both.
  2. ICA artifact cleaning (infomax + ICLabel): run it (a cache in
     clean_data_*/ is reused) or skip it.
  3. Correlation analysis: single experiment, pooled, both, or none.
  4. Machine-learning models: all at once, or pick any of Logistic,
     Logistic + ANOVA, Linear, Neural Network (3 architectures), HMM.
  5. Optional thinking-window and phase analyses (only with ICA).

Every model is evaluated leave-one-group-out: each participant label is
held out in turn and the scaler, feature selector, threshold, HMM state
mapping and network are fit on the remaining labels only. Labels identify
recordings ("BrainSoft Participant N"), so this is effectively
leave-one-recording-out. Plots and CSVs are written to
attention_results/ and cognitive_load_results/.
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quiet TensorFlow info/warning spam
import sys
import warnings
import json
import re
import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from scipy.special import expit

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_curve
from sklearn.decomposition import PCA

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

try:
    import hmmlearn.hmm as hmm
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False

mne.set_log_level("WARNING")

# ============================================================
# CONFIG
# ============================================================
# The anonymized recordings live in a "data" folder next to this script
# (paper_final_code/data/...). If you instead unpack the four data folders
# directly beside the script, that works too: whichever location holds
# pb_attention/ is used.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_data_root() -> str:
    for cand in (os.path.join(BASE_DIR, "data"), BASE_DIR):
        if os.path.isdir(os.path.join(cand, "pb_attention")):
            return cand
    return os.path.join(BASE_DIR, "data")


DATA_ROOT = _find_data_root()
PB_DIR = {"attention": f"{DATA_ROOT}/pb_attention", "memory": f"{DATA_ROOT}/pb_cognitive_load"}
EEG_DIR = {"attention": f"{DATA_ROOT}/eeg_attention", "memory": f"{DATA_ROOT}/eeg_cognitive_load"}
PARTICIPANTS_CSV = f"{DATA_ROOT}/participants.csv"
RESULTS_DIR = {"attention": "./attention_results", "memory": "./cognitive_load_results"}
CLEAN_DATA_DIR = {"attention": "./clean_data_attention", "memory": "./clean_data_memory"}
CLEAN_EEG_PB_DIR = "./Clean_Eeg_using_ICA"

# --- Reviewer comment 2: click-locked thinking/decision/motor windows ---
# A trial's click time is reconstructed from the task's own event log
# (match_times/latency_times/false_positives), rescaled onto the EEG's
# own timeline (see rescale_task_times_to_eeg below), then:
#   thinking window:  [stimulus_onset, click_time - DEFAULT_BUFFER_MS)
#   motor window:     [click_time - DEFAULT_BUFFER_MS, click_time + DEFAULT_BUFFER_MS)
# This replaces averaging power bands over the WHOLE inter-stimulus
# interval (which mixes pre-click "thinking" activity with click-locked
# motor-execution artifacts) with features computed only in the window
# that actually precedes/follows the click.
DEFAULT_BUFFER_MS = 200

# Fine-grained stride used ONLY for the thinking-vs-decision phase split
# below (Step 6c) -- at the pipeline's normal STRIDE=16 (125 ms/window) a
# typical ~400 ms hit reaction time contains just 3-7 power-band rows,
# too few for any within-trial segmentation. Recomputed directly from the
# already-cached cleaned EEG signal (CLEAN_EEG_PB_DIR/*_cleaned_eeg.csv),
# so no extra ICA run is needed.
FINE_STRIDE = 4

SFREQ = 128.0
WINDOW_SIZE = 256
STRIDE = 16
ADC_TO_UV = 0.51

EMOTIV_CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2",
                    "P8", "T8", "FC6", "F4", "F8", "AF4"]
BANDS = {"delta": (0.5, 4), "theta": (4, 7), "alpha": (7, 14),
          "beta": (14, 30), "gamma": (30, 45)}
PB_VARIANTS = ["alpha", "beta", "delta", "gamma", "theta"]
BCI_ELECTRODES = ["af3", "af4", "f3", "f4", "f7", "f8", "fc5", "fc6",
                    "o1", "o2", "p7", "p8", "t7", "t8"]


# ============================================================
# STEP 1: Load the experiments that ship with this code
# ============================================================
# The (anonymized) recordings are distributed alongside the code in the
# same folder it is run from -- nothing is downloaded. Expected layout:
#   pb_attention/{id}-attention.json   pb_attention/{id}-pb.json
#   eeg_attention/{id}-attention-eeg.json
#   pb_cognitive_load/{id}-memory.json pb_cognitive_load/{id}-pb.json
#   eeg_cognitive_load/{id}-memory-eeg.json

def discover_local_experiments(kinds=("attention", "memory")) -> dict:
    """Experiment IDs available on disk for the requested task(s), found by
    scanning the data folders for complete sets (label JSON + power-band
    JSON + raw EEG JSON). Returns {kind: [ids]}."""
    print(f"\n{'#'*70}\nLoading the local, anonymized dataset from {DATA_ROOT}/\n{'#'*70}")
    ids_by_kind = {}
    for kind in kinds:
        pb_dir, eeg_dir = PB_DIR[kind], EEG_DIR[kind]
        if not os.path.isdir(pb_dir) or not os.path.isdir(eeg_dir):
            raise SystemExit(f"\nData folders {pb_dir}/ and {eeg_dir}/ not found.\n"
                              f"Place the dataset in a 'data' folder next to this script (see README.md).")
        ids = []
        for fname in os.listdir(pb_dir):
            m = re.fullmatch(rf"(\d+)-{kind}\.json", fname)
            if not m:
                continue
            i = int(m.group(1))
            if os.path.exists(f"{pb_dir}/{i}-pb.json") and os.path.exists(f"{eeg_dir}/{i}-{kind}-eeg.json"):
                ids.append(i)
        ids_by_kind[kind] = sorted(ids)
        print(f"{'attention' if kind == 'attention' else 'cognitive load'}: {len(ids)} experiments found")
    return ids_by_kind


# ============================================================
# STEP 2: ICA cleaning -- SKIPPED if a cached cleaned dataset exists
# ============================================================

def load_eeg_export(path: str) -> mne.io.RawArray:
    data = json.load(open(path))
    header, rows = data[0], data[1:]
    rec = dict(zip(header, rows[0]))
    flat = np.array(rec["data"], dtype=float)
    n_channels = len(EMOTIV_CHANNELS)
    n_timesteps = len(flat) // n_channels
    reshaped = flat[:n_timesteps * n_channels].reshape(n_timesteps, n_channels).T
    data_centered = (reshaped - reshaped.mean(axis=1, keepdims=True)) * ADC_TO_UV
    info = mne.create_info(ch_names=EMOTIV_CHANNELS, sfreq=SFREQ, ch_types="eeg")
    raw = mne.io.RawArray(data_centered * 1e-6, info)
    raw.set_montage("standard_1020", match_case=False, on_missing="warn")
    return raw


def clean_raw(raw: mne.io.RawArray, n_components: int = 10) -> dict:
    """Returns filtered-but-uncleaned raw, cleaned raw, and excluded
    component labels -- everything needed both for the per-trial
    dataset AND for the aggregate ICA diagnostic plots (Step 5)."""
    raw_f = raw.copy().filter(l_freq=1.0, h_freq=45.0, fir_design="firwin")
    raw_f.set_eeg_reference("average")
    ica = ICA(n_components=n_components, method="infomax", fit_params=dict(extended=True),
               random_state=42, max_iter="auto")
    ica.fit(raw_f)
    labels = label_components(raw_f, ica, method="iclabel")["labels"]
    exclude_idx = [i for i, l in enumerate(labels) if l not in ("brain", "other")]
    ica.exclude = exclude_idx
    raw_clean = raw_f.copy()
    ica.apply(raw_clean)
    return {"raw_filtered": raw_f, "raw_clean": raw_clean,
             "excluded_labels": [labels[i] for i in exclude_idx]}


def sliding_window_power_bands(raw: mne.io.RawArray) -> pd.DataFrame:
    data = raw.get_data() * 1e6
    n_channels, n_total = data.shape
    n_windows = (n_total - WINDOW_SIZE) // STRIDE + 1
    freqs = np.fft.rfftfreq(WINDOW_SIZE, d=1.0 / SFREQ)
    hann = np.hanning(WINDOW_SIZE)
    band_idx = {b: np.logical_and(freqs >= lo, freqs < hi) for b, (lo, hi) in BANDS.items()}
    rows = []
    for w in range(n_windows):
        start = w * STRIDE
        window = data[:, start:start + WINDOW_SIZE]
        row = {"subindex": w + 1}
        for ci, ch in enumerate(EMOTIV_CHANNELS):
            x = window[ci]
            detrended = x - np.polyval(np.polyfit(np.arange(len(x)), x, 1), np.arange(len(x)))
            power = np.abs(np.fft.rfft(detrended * hann)) ** 2
            for band in BANDS:
                row[f"{band}_{ch.lower()}"] = power[band_idx[band]].sum()
        rows.append(row)
    return pd.DataFrame(rows)


def block_average(df: pd.DataFrame, n_blocks: int):
    n_rows = len(df)
    block_size = n_rows // n_blocks
    if block_size == 0:
        return None
    feature_cols = [c for c in df.columns if c != "subindex"]
    trimmed = df[feature_cols].iloc[:block_size * n_blocks]
    grouped = trimmed.groupby(np.arange(len(trimmed)) // block_size).mean()
    return grouped.iloc[:n_blocks].reset_index(drop=True)


def compute_band_power_single_window(raw: mne.io.RawArray, window_sec: float = 2.0) -> dict:
    """Single-window Welch PSD, used only for the aggregate per-channel
    before/after comparison (Step 5), not the per-trial features."""
    spectrum = raw.compute_psd(method="welch", fmin=0.5, fmax=45, n_fft=int(window_sec * raw.info["sfreq"]))
    psds, freqs = spectrum.get_data(return_freqs=True)
    out = {}
    for band, (lo, hi) in BANDS.items():
        idx = np.logical_and(freqs >= lo, freqs < hi)
        for ci, ch in enumerate(raw.ch_names):
            out[f"{band}_{ch.lower()}"] = psds[ci, idx].sum()
    return out


def load_pb(path: str) -> pd.DataFrame:
    data = json.load(open(path))
    df = pd.DataFrame(data[1:], columns=data[0])
    feature_cols = [c for c in df.columns if c not in ("subindex", "timestamp")]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    return df.sort_values("subindex").reset_index(drop=True)


def load_labels(path: str) -> dict:
    data = json.load(open(path))
    return dict(zip(data[0], data[1]))


_PARTICIPANT_MAP = None


def participant_label(kind: str, exp_id) -> str:
    """Anonymized participant label ("BrainSoft Participant N") for an
    experiment, from participants.csv. The data files themselves contain
    NO names; this table is the only place a recording is tied to a
    participant label, and it is what groups trials for leave-one-subject-out."""
    global _PARTICIPANT_MAP
    if _PARTICIPANT_MAP is None:
        if not os.path.exists(PARTICIPANTS_CSV):
            raise SystemExit(f"\n{PARTICIPANTS_CSV} not found -- it ships with the data (see README.md).")
        df = pd.read_csv(PARTICIPANTS_CSV)
        _PARTICIPANT_MAP = {(("memory" if t == "cognitive_load" else t), int(i)): p
                            for p, t, i in zip(df["participant"], df["task"], df["experiment_id"])}
    return _PARTICIPANT_MAP[(kind, int(exp_id))]


def save_cleaned_eeg_data(raw_clean: mne.io.RawArray, exp_id, kind: str):
    """Saves the actual CLEANED EEG SIGNAL itself (post-ICA, the raw
    time series -- not a power-band conversion of it) to disk. One row
    per sample, one column per channel, in microvolts."""
    os.makedirs(CLEAN_EEG_PB_DIR, exist_ok=True)
    data_uv = raw_clean.get_data() * 1e6  # volts -> microvolts, channels x samples
    df = pd.DataFrame(data_uv.T, columns=raw_clean.ch_names)
    df.insert(0, "sample_index", range(len(df)))
    out_path = f"{CLEAN_EEG_PB_DIR}/{exp_id}_{kind}_cleaned_eeg.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved cleaned EEG signal -> {out_path}")


def save_full_resolution_cleaned_pb(cleaned_pb: pd.DataFrame, exp_id, kind: str):
    """Saves the FULL-resolution power-band table (one row per sliding
    window, NOT block-averaged to one row per trial) computed from the
    ICA-cleaned EEG signal -- a standalone record of exactly what
    'clean the EEG, then convert to power bands' produced for this
    experiment, independent of anything else in the pipeline."""
    os.makedirs(CLEAN_EEG_PB_DIR, exist_ok=True)
    out_path = f"{CLEAN_EEG_PB_DIR}/{exp_id}_{kind}_cleaned_pb.csv"
    cleaned_pb.to_csv(out_path, index=False)
    print(f"  Saved full-resolution cleaned power bands -> {out_path}")


def load_cached_clean_data(kind: str):
    """Returns (X, y, subjects, exp_ids) if a cached cleaned dataset
    exists on disk, else None."""
    path = f"{CLEAN_DATA_DIR[kind]}/cleaned_power_bands.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "exp_id" not in df.columns:
        return None  # older cache format without exp_id -- treat as missing, rebuild
    y = df["Sucess"].to_numpy()
    subjects = df["subject"].to_numpy()
    exp_ids = df["exp_id"].to_numpy()
    X = df.drop(columns=["Sucess", "subject", "exp_id"])
    return X, y, subjects, exp_ids


def save_ica_diagnostics(ica_diagnostics: list, kind: str):
    """Persist per-experiment ICA diagnostics next to the cleaned-data
    cache, so the Step 5 plots (component exclusion, channel effect) can
    be regenerated on later runs that skip ICA via the cache."""
    os.makedirs(CLEAN_DATA_DIR[kind], exist_ok=True)
    with open(f"{CLEAN_DATA_DIR[kind]}/ica_diagnostics.json", "w") as f:
        json.dump(ica_diagnostics, f, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def load_ica_diagnostics(kind: str):
    path = f"{CLEAN_DATA_DIR[kind]}/ica_diagnostics.json"
    if not os.path.exists(path):
        return None
    try:
        return json.load(open(path))
    except Exception:
        return None


def build_cleaned_and_uncleaned(ids: list, kind: str) -> dict:
    """Builds cleaned + uncleaned datasets AND collects the ICA
    diagnostics needed for Step 5's aggregate plots. If a cached
    cleaned dataset already exists, ICA is skipped entirely and the
    cache is loaded directly -- ICA diagnostics will be unavailable in
    that case (they are only produced on a fresh run)."""
    cached = load_cached_clean_data(kind)
    if cached is not None:
        print(f"\n{'#'*70}\nSTEP 2: Cached cleaned dataset found for {kind.upper()} -- "
              f"SKIPPING ICA entirely\n{'#'*70}")
        X_clean, y, subjects, exp_ids = cached
        print(f"Loaded {len(X_clean)} cached rows from {CLEAN_DATA_DIR[kind]}/cleaned_power_bands.csv")

        # Uncleaned still needs to be (re)built for the clean-vs-uncleaned
        # comparison -- this does not require ICA, so it stays cheap. BUT
        # it must be built from the SAME experiments, in the SAME order,
        # as the cached cleaned dataset -- not the current `ids` list from
        # discover_local_experiments(). If the data folder has gained/lost
        # experiments since the cache was built, those two lists diverge,
        # X_uncleaned ends up a different length than y (cleaned), and
        # run_models_on_version's LOSO indexing crashes on mismatched
        # sample counts. dict.fromkeys preserves first-seen order so row
        # order matches X_clean's experiment-by-experiment layout exactly.
        cached_ids = list(dict.fromkeys(exp_ids.tolist()))
        X_uncleaned, y_u, subjects_u, exp_ids_u = build_uncleaned_only(cached_ids, kind)

        if len(X_uncleaned) != len(X_clean):
            print(f"  [warn] uncleaned dataset has {len(X_uncleaned)} rows vs. {len(X_clean)} cleaned rows "
                  f"even after restricting to the cache's own experiment IDs -- some of those experiments' "
                  f"raw power-band files may be missing or short. Falling back to uncleaned == cleaned's "
                  f"experiment set is not possible here; the clean-vs-uncleaned comparison for {kind} will "
                  f"use the uncleaned data's own exp_ids/y instead of the cleaned ones to avoid a crash, "
                  f"but the two are no longer guaranteed to be row-for-row the same trials.")
            return {
                "cleaned": (X_clean, y, subjects, exp_ids),
                "uncleaned": (X_uncleaned, y_u, subjects_u, exp_ids_u),
                "ica_diagnostics": load_ica_diagnostics(kind),
                "uncleaned_y_mismatch": True,
            }

        return {
            "cleaned": (X_clean, y, subjects, exp_ids),
            "uncleaned": (X_uncleaned, y_u, subjects_u, exp_ids_u),
            "ica_diagnostics": load_ica_diagnostics(kind),  # None if cache predates this feature
        }

    print(f"\n{'#'*70}\nSTEP 2: ICA cleaning -- {kind.upper()}\n{'#'*70}")
    pb_dir, eeg_dir = PB_DIR[kind], EEG_DIR[kind]
    cleaned_rows, uncleaned_rows, ys, subjects, exp_ids = [], [], [], [], []
    ica_diagnostics = []  # for Step 5's aggregate plots

    for exp_id in ids:
        label_path, pb_path = f"{pb_dir}/{exp_id}-{kind}.json", f"{pb_dir}/{exp_id}-pb.json"
        eeg_path = f"{eeg_dir}/{exp_id}-{kind}-eeg.json"
        if not (os.path.exists(label_path) and os.path.exists(pb_path) and os.path.exists(eeg_path)):
            continue
        try:
            rec = load_labels(label_path)
            detect_success = np.array(rec["detectSuccess"])
            subject = participant_label(kind, exp_id)
            n_trials = len(detect_success)

            uncleaned_pb = load_pb(pb_path)
            feature_cols = [c for c in uncleaned_pb.columns if c not in ("subindex", "timestamp")]
            uncleaned_avg = block_average(uncleaned_pb[feature_cols + ["subindex"]], n_trials)

            raw = load_eeg_export(eeg_path)
            clean_result = clean_raw(raw)
            save_cleaned_eeg_data(clean_result["raw_clean"], exp_id, kind)
            cleaned_pb = sliding_window_power_bands(clean_result["raw_clean"])
            save_full_resolution_cleaned_pb(cleaned_pb, exp_id, kind)
            cleaned_avg = block_average(cleaned_pb, n_trials)

            if uncleaned_avg is None or cleaned_avg is None or \
               len(uncleaned_avg) != n_trials or len(cleaned_avg) != n_trials:
                print(f"  [skip] experiment {exp_id}: block-averaging mismatch (recording too short)")
                continue

            # --- ICA diagnostics for Step 5 (per-channel before/after) ---
            pb_before = compute_band_power_single_window(clean_result["raw_filtered"])
            pb_after = compute_band_power_single_window(clean_result["raw_clean"])
            channel_pct_change = {}
            for ch in EMOTIV_CHANNELS:
                pcts = []
                for band in BANDS:
                    key = f"{band}_{ch.lower()}"
                    b, a = pb_before[key], pb_after[key]
                    pcts.append(100 * (a - b) / b if b != 0 else np.nan)
                channel_pct_change[ch] = float(np.nanmean(pcts))
            ica_diagnostics.append({"exp_id": exp_id, "excluded_labels": clean_result["excluded_labels"],
                                      **channel_pct_change})

        except Exception as e:
            print(f"  [error] experiment {exp_id}: {e}")
            continue

        cleaned_rows.append(cleaned_avg)
        uncleaned_rows.append(uncleaned_avg)
        ys.append(detect_success)
        subjects.append(np.full(n_trials, subject))
        exp_ids.append(np.full(n_trials, exp_id))
        print(f"  exp {exp_id} ({subject}): {n_trials} trials, "
              f"{len(clean_result['excluded_labels'])} ICA component(s) removed")

    if not cleaned_rows:
        empty = (pd.DataFrame(), np.array([]), np.array([]), np.array([]))
        return {"cleaned": empty, "uncleaned": empty, "ica_diagnostics": []}

    X_clean = pd.concat(cleaned_rows, ignore_index=True)
    X_uncleaned = pd.concat(uncleaned_rows, ignore_index=True)
    y = np.concatenate(ys)
    subjects = np.concatenate(subjects)
    exp_ids = np.concatenate(exp_ids)
    save_ica_diagnostics(ica_diagnostics, kind)
    return {
        "cleaned": (X_clean, y, subjects, exp_ids),
        "uncleaned": (X_uncleaned, y, subjects, exp_ids),
        "ica_diagnostics": ica_diagnostics,
    }


def build_uncleaned_only(ids: list, kind: str) -> tuple:
    """Used when the cleaned cache is hit -- still need an uncleaned
    version for the clean-vs-uncleaned comparison, but this is cheap
    (no ICA required)."""
    pb_dir = PB_DIR[kind]
    rows, ys, subjects, exp_ids = [], [], [], []
    for exp_id in ids:
        label_path, pb_path = f"{pb_dir}/{exp_id}-{kind}.json", f"{pb_dir}/{exp_id}-pb.json"
        if not (os.path.exists(label_path) and os.path.exists(pb_path)):
            continue
        try:
            rec = load_labels(label_path)
            detect_success = np.array(rec["detectSuccess"])
            subject = participant_label(kind, exp_id)
            n_trials = len(detect_success)
            pb_df = load_pb(pb_path)
            feature_cols = [c for c in pb_df.columns if c not in ("subindex", "timestamp")]
            avg = block_average(pb_df[feature_cols + ["subindex"]], n_trials)
            if avg is None or len(avg) != n_trials:
                continue
        except Exception:
            continue
        rows.append(avg)
        ys.append(detect_success)
        subjects.append(np.full(n_trials, subject))
        exp_ids.append(np.full(n_trials, exp_id))
    if not rows:
        return pd.DataFrame(), np.array([]), np.array([]), np.array([])
    return (pd.concat(rows, ignore_index=True), np.concatenate(ys),
             np.concatenate(subjects), np.concatenate(exp_ids))


# ============================================================
# STEP 2b: click-locked thinking / decision / motor windows
# (reviewer comment 2) -- built on top of the full-resolution cleaned
# power bands already saved by Step 2 (CLEAN_EEG_PB_DIR/*_cleaned_pb.csv)
# and, for the finer-grained phase split, the cleaned EEG signal itself
# (CLEAN_EEG_PB_DIR/*_cleaned_eeg.csv). Neither needs a fresh ICA run --
# both files exist on disk after Step 2 has run at least once, cached or
# not -- so this works whether or not Step 2 hit its cache this session.
# ============================================================

def rescale_task_times_to_eeg(match_times: list, latency_times: list,
                                 false_positives: list, detect_success: list,
                                 eeg_duration_ms: float) -> dict:
    """match_times/latency_times/false_positives are on the task
    software's own internal clock, not the EEG recording's clock -- the
    two don't share a zero-point or necessarily the same rate. This
    stretches the task's own timeline to fit the EEG's measured
    duration, preserving relative event order/spacing."""
    all_raw_times = list(match_times) + list(false_positives)
    hit_click_times_raw = []
    hit_counter = 0
    for i, t in enumerate(match_times):
        if i < len(detect_success) and detect_success[i] == 1 and hit_counter < len(latency_times):
            hit_click_times_raw.append(t + latency_times[hit_counter])
            hit_counter += 1
    all_raw_times += hit_click_times_raw

    task_duration_ms = max(all_raw_times) if all_raw_times else eeg_duration_ms
    scale_factor = eeg_duration_ms / task_duration_ms if task_duration_ms > 0 else 1.0

    return {
        "match_times": [t * scale_factor for t in match_times],
        "hit_click_times": [t * scale_factor for t in hit_click_times_raw],
        "false_positives": [t * scale_factor for t in false_positives],
        "scale_factor": scale_factor,
        "task_duration_ms": task_duration_ms,
        "eeg_duration_ms": eeg_duration_ms,
    }


def reconstruct_all_trials_from_rescaled(rescaled: dict, match_times: list, detect_success: list) -> list:
    """Every trial (hits AND misses), tagged with its outcome. Misses
    carry click_time_ms=None -- there was no click to reconstruct."""
    trials = []
    hit_click_idx = 0
    for i in range(len(match_times)):
        is_hit = i < len(detect_success) and detect_success[i] == 1
        stim_onset = rescaled["match_times"][i]
        if is_hit and hit_click_idx < len(rescaled["hit_click_times"]):
            click_time = rescaled["hit_click_times"][hit_click_idx]
            hit_click_idx += 1
            trials.append({"trial_idx": i, "outcome": 1, "stimulus_onset_ms": stim_onset,
                             "click_time_ms": click_time})
        else:
            trials.append({"trial_idx": i, "outcome": 0, "stimulus_onset_ms": stim_onset,
                             "click_time_ms": None})
    return trials


def _ms_to_window_index(time_ms: float, stride: int = STRIDE) -> int:
    window_step_ms = (stride / SFREQ) * 1000
    return int(round(time_ms / window_step_ms))


def extract_window_features(pb_table: pd.DataFrame, start_ms: float, end_ms: float,
                              stride: int = STRIDE) -> pd.Series:
    """Averages sliding-window power-band rows whose window falls within
    [start_ms, end_ms). pb_table must have a 'subindex' column (1-indexed
    window number) alongside the feature columns."""
    w_lo = _ms_to_window_index(start_ms, stride)
    w_hi = _ms_to_window_index(end_ms, stride)
    feature_cols = [c for c in pb_table.columns if c not in ("subindex", "center_ms")]
    subset = pb_table[(pb_table["subindex"] - 1 >= w_lo) & (pb_table["subindex"] - 1 < w_hi)]
    if len(subset) == 0:
        nearest_idx = (pb_table["subindex"] - 1 - w_lo).abs().idxmin()
        return pb_table.loc[nearest_idx, feature_cols]
    return subset[feature_cols].mean()


def build_thinking_only_dataset_for_kind(ids: list, kind: str, buffer_ms: float = DEFAULT_BUFFER_MS) -> tuple:
    """Builds a THINKING-WINDOW-ONLY feature dataset ([stimulus_onset,
    click_time - buffer_ms)) covering every trial (hits + misses) of
    every experiment in `ids`, answering reviewer comment 2's concern
    that motor-execution artifacts at the click moment contaminate
    whole-trial-averaged features. Misses have no real click -- their
    thinking-window end is ESTIMATED as stimulus_onset + that
    experiment's own median hit reaction time (300 ms if the experiment
    has no hits at all), flagged via the returned is_estimated array.
    Requires Step 2 to have run at least once for each experiment (reads
    its cached full-resolution cleaned power bands and label JSON --
    no fresh ICA needed here)."""
    pb_dir = PB_DIR[kind]
    rows, ys, exp_id_list, is_estimated_list = [], [], [], []
    for exp_id in ids:
        label_path = f"{pb_dir}/{exp_id}-{kind}.json"
        cleaned_pb_path = f"{CLEAN_EEG_PB_DIR}/{exp_id}_{kind}_cleaned_pb.csv"
        if not (os.path.exists(label_path) and os.path.exists(cleaned_pb_path)):
            continue
        try:
            rec = load_labels(label_path)
            match_times = rec["match_times"]
            latency_times = rec.get("latency_times") or []
            false_positives = rec.get("false_positives") or []
            detect_success = rec["detectSuccess"]
            cleaned_pb = pd.read_csv(cleaned_pb_path)

            window_step_ms = (STRIDE / SFREQ) * 1000
            eeg_duration_ms = cleaned_pb["subindex"].max() * window_step_ms + (WINDOW_SIZE / SFREQ) * 1000

            rescaled = rescale_task_times_to_eeg(match_times, latency_times, false_positives,
                                                   detect_success, eeg_duration_ms)
            trials = reconstruct_all_trials_from_rescaled(rescaled, match_times, detect_success)

            hit_rts = [t["click_time_ms"] - t["stimulus_onset_ms"] for t in trials
                        if t["outcome"] == 1]
            fallback_rt = float(np.median(hit_rts)) if hit_rts else 300.0

            for t in trials:
                is_est = t["click_time_ms"] is None
                click_time = t["click_time_ms"] if not is_est else t["stimulus_onset_ms"] + fallback_rt
                feats = extract_window_features(cleaned_pb, t["stimulus_onset_ms"],
                                                  max(click_time - buffer_ms, t["stimulus_onset_ms"] + 1))
                rows.append(feats)
                ys.append(t["outcome"])
                exp_id_list.append(exp_id)
                is_estimated_list.append(is_est)
        except Exception as e:
            print(f"  [error] thinking-only window, experiment {exp_id}: {e}")
            continue
    if not rows:
        return pd.DataFrame(), np.array([]), np.array([]), np.array([])
    X = pd.DataFrame(rows).reset_index(drop=True)
    return X, np.array(ys), np.array(exp_id_list), np.array(is_estimated_list)


def sliding_window_power_bands_finer(data: np.ndarray, stride: int) -> pd.DataFrame:
    """Same FFT/band-power logic as sliding_window_power_bands(), but
    operating directly on a (channels x samples) microvolt array -- so it
    can be recomputed with a smaller stride directly from the cached
    cleaned EEG signal, without re-running ICA. Adds a 'center_ms' column
    (the window's center sample, in ms) used to assign each row a
    position in time, both absolute and relative-within-trial."""
    n_channels, n_total = data.shape
    n_windows = (n_total - WINDOW_SIZE) // stride + 1
    freqs = np.fft.rfftfreq(WINDOW_SIZE, d=1.0 / SFREQ)
    hann = np.hanning(WINDOW_SIZE)
    band_idx = {b: np.logical_and(freqs >= lo, freqs < hi) for b, (lo, hi) in BANDS.items()}
    rows, centers = [], []
    for w in range(n_windows):
        start = w * stride
        window = data[:, start:start + WINDOW_SIZE]
        row = {"subindex": w + 1}
        for ci, ch in enumerate(EMOTIV_CHANNELS):
            x = window[ci]
            detrended = x - np.polyval(np.polyfit(np.arange(len(x)), x, 1), np.arange(len(x)))
            power = np.abs(np.fft.rfft(detrended * hann)) ** 2
            for band in BANDS:
                row[f"{band}_{ch.lower()}"] = power[band_idx[band]].sum()
        rows.append(row)
        centers.append((start + WINDOW_SIZE / 2) / SFREQ * 1000)
    df = pd.DataFrame(rows)
    df["center_ms"] = centers
    return df


def best_split_point(seq: np.ndarray) -> int:
    """Finds the single change point k (2 <= k <= L-2) that minimizes
    total within-segment sum-of-squares across [0,k) and [k,L) -- the
    standard optimal binary mean-shift segmentation, computed in O(L)
    via cumulative sums. Returns None if the sequence is too short
    (fewer than 4 rows) to split meaningfully. This is what actually
    classifies a trial's own power-band trajectory into a "thinking"
    sub-phase and a "decision" sub-phase: the point in THAT trial where
    its power-band pattern shifts the most, not an assumed fixed split."""
    L = seq.shape[0]
    if L < 4:
        return None
    cumsum = np.cumsum(seq, axis=0)
    cumsum2 = np.cumsum(seq ** 2, axis=0)
    total_sum, total_sum2 = cumsum[-1], cumsum2[-1]
    best_k, best_cost = None, np.inf
    for k in range(2, L - 1):
        n1 = k
        s1, s2_ = cumsum[k - 1], cumsum2[k - 1]
        var1 = (s2_ - s1 ** 2 / n1).sum()
        n2 = L - k
        s1b, s2b = total_sum - s1, total_sum2 - s2_
        var2 = (s2b - s1b ** 2 / n2).sum()
        cost = var1 + var2
        if cost < best_cost:
            best_cost, best_k = cost, k
    return best_k


def classify_phases_single_experiment(exp_id, kind: str, buffer_ms: float = DEFAULT_BUFFER_MS,
                                         fine_stride: int = FINE_STRIDE) -> dict:
    """Data-driven thinking/decision/motor phase classification for one
    experiment. For every HIT trial (only hits have a real, non-estimated
    click time -- see note below), power bands are recomputed at a finer
    stride directly from the cached cleaned EEG signal, giving each trial
    roughly 4x more rows than the pipeline's normal 125 ms resolution.
    Within [stimulus_onset, click_time), best_split_point() finds the
    genuine change point in THAT trial's own (PCA-reduced) trajectory,
    splitting it into "thinking" (before) and "decision" (after); the
    already-validated click-locked "motor" window
    ([click_time - buffer_ms, click_time + buffer_ms)) follows.

    Misses are included too, using the estimated click time (same
    fallback as build_thinking_only_dataset_for_kind), flagged via
    is_estimated -- the change-point method only needs *a* window
    boundary to split, not a true one, though the split is less
    trustworthy for these.

    Returns per-trial boundaries plus, for every trial, the mean
    power-band feature vector within each of the three phases (for
    downstream "which phase best predicts hit/miss" modelling)."""
    pb_dir = PB_DIR[kind]
    label_path = f"{pb_dir}/{exp_id}-{kind}.json"
    eeg_path = f"{CLEAN_EEG_PB_DIR}/{exp_id}_{kind}_cleaned_eeg.csv"
    if not (os.path.exists(label_path) and os.path.exists(eeg_path)):
        return {"trials": pd.DataFrame(), "phase_features": pd.DataFrame()}

    rec = load_labels(label_path)
    match_times, latency_times = rec["match_times"], rec.get("latency_times") or []
    false_positives, detect_success = rec.get("false_positives") or [], rec["detectSuccess"]

    eeg_df = pd.read_csv(eeg_path)
    data = eeg_df[EMOTIV_CHANNELS].values.T
    eeg_duration_ms = data.shape[1] / SFREQ * 1000
    fine_pb = sliding_window_power_bands_finer(data, fine_stride)
    feature_cols = [c for c in fine_pb.columns if c not in ("subindex", "center_ms")]

    rescaled = rescale_task_times_to_eeg(match_times, latency_times, false_positives,
                                           detect_success, eeg_duration_ms)
    trials = reconstruct_all_trials_from_rescaled(rescaled, match_times, detect_success)
    hit_rts = [t["click_time_ms"] - t["stimulus_onset_ms"] for t in trials if t["outcome"] == 1]
    fallback_rt = float(np.median(hit_rts)) if hit_rts else 300.0

    # Fit one PCA per experiment, across all trials' thinking-window rows,
    # so every trial's change point is found in the same reduced space.
    per_trial_idx, per_trial_rows = [], []
    for t in trials:
        is_est = t["click_time_ms"] is None
        click_time = t["click_time_ms"] if not is_est else t["stimulus_onset_ms"] + fallback_rt
        mask = (fine_pb["center_ms"] >= t["stimulus_onset_ms"]) & (fine_pb["center_ms"] < click_time)
        sub = fine_pb.loc[mask, feature_cols + ["center_ms"]]
        if len(sub) < 4:
            continue
        per_trial_idx.append(t["trial_idx"])
        per_trial_rows.append(sub)

    if not per_trial_rows:
        return {"trials": pd.DataFrame(), "phase_features": pd.DataFrame()}

    X_all = np.vstack([s[feature_cols].values for s in per_trial_rows])
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X_all)
    n_comp = min(8, Xz.shape[1], Xz.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=42)
    Xp = pca.fit_transform(Xz)

    trial_boundary_rows, phase_feature_rows = [], []
    idx = 0
    for ti, sub in zip(per_trial_idx, per_trial_rows):
        L = len(sub)
        seg = Xp[idx:idx + L]
        idx += L
        t = trials[ti]
        is_est = t["click_time_ms"] is None
        click_time = t["click_time_ms"] if not is_est else t["stimulus_onset_ms"] + fallback_rt
        stim = t["stimulus_onset_ms"]

        k = best_split_point(seg)
        if k is None:
            decision_split_ms = stim + 0.5 * (click_time - stim)  # too short to split; midpoint fallback
        else:
            decision_split_ms = float(sub["center_ms"].iloc[k - 1])

        trial_boundary_rows.append({
            "trial_idx": ti, "outcome": t["outcome"], "is_estimated_click": is_est,
            "stimulus_onset_ms": stim, "decision_split_ms": decision_split_ms,
            "click_time_ms": click_time, "reaction_time_ms": click_time - stim,
            "decision_split_frac": (decision_split_ms - stim) / max(click_time - stim, 1e-6),
        })

        thinking_feats = extract_window_features(fine_pb, stim, decision_split_ms, stride=fine_stride)
        decision_feats = extract_window_features(fine_pb, decision_split_ms, click_time, stride=fine_stride)
        motor_feats = extract_window_features(fine_pb, click_time - buffer_ms, click_time + buffer_ms,
                                                 stride=fine_stride)
        phase_row = {"trial_idx": ti, "outcome": t["outcome"]}
        for c in feature_cols:
            phase_row[f"thinking_{c}"] = thinking_feats[c]
            phase_row[f"decision_{c}"] = decision_feats[c]
            phase_row[f"motor_{c}"] = motor_feats[c]
        phase_feature_rows.append(phase_row)

    return {"trials": pd.DataFrame(trial_boundary_rows),
             "phase_features": pd.DataFrame(phase_feature_rows)}


def build_phase_averaged_dataset_for_kind(ids: list, kind: str) -> dict:
    """Runs classify_phases_single_experiment() over every experiment in
    `ids` and concatenates the per-trial phase-averaged feature tables,
    producing three same-shaped feature sets (thinking-only, decision-only,
    motor-only -- all click-locked and data-driven) plus the shared
    outcome labels, so the existing model suite can be run on each and
    compared against the whole-trial result."""
    all_trials, all_phase = [], []
    for exp_id in ids:
        result = classify_phases_single_experiment(exp_id, kind)
        if len(result["trials"]) == 0:
            continue
        result["trials"]["exp_id"] = exp_id
        result["phase_features"]["exp_id"] = exp_id
        all_trials.append(result["trials"])
        all_phase.append(result["phase_features"])
    if not all_phase:
        return {"trial_boundaries": pd.DataFrame(), "thinking": (pd.DataFrame(), np.array([])),
                 "decision": (pd.DataFrame(), np.array([])), "motor": (pd.DataFrame(), np.array([])),
                 "exp_ids": np.array([])}

    trial_boundaries = pd.concat(all_trials, ignore_index=True)
    phase_features = pd.concat(all_phase, ignore_index=True)
    y = phase_features["outcome"].to_numpy()
    exp_ids = phase_features["exp_id"].to_numpy()

    def _subset(prefix):
        cols = [c for c in phase_features.columns if c.startswith(prefix + "_")]
        X = phase_features[cols].copy()
        X.columns = [c[len(prefix) + 1:] for c in X.columns]
        return X, y

    return {"trial_boundaries": trial_boundaries,
             "thinking": _subset("thinking"), "decision": _subset("decision"),
             "motor": _subset("motor"), "exp_ids": exp_ids}


def plot_phase_timeline_single_experiment(trial_boundaries: pd.DataFrame, exp_id, kind: str, out_dir: str):
    """Horizontal stacked timeline: one bar per trial, split into its
    thinking / decision / motor segments (in ms since stimulus onset),
    for the one chosen experiment -- a visual check of what the
    change-point method actually found."""
    df = trial_boundaries[trial_boundaries["exp_id"] == exp_id].sort_values("trial_idx") \
        if "exp_id" in trial_boundaries.columns else trial_boundaries.sort_values("trial_idx")
    if len(df) == 0:
        return
    fig, ax = plt.subplots(figsize=(10, max(3, 0.3 * len(df))))
    colors = {"thinking": "#4C72B0", "decision": "#DD8452", "motor": "#C44E52"}
    for row_i, (_, r) in enumerate(df.iterrows()):
        thinking_dur = r["decision_split_ms"] - r["stimulus_onset_ms"]
        decision_dur = r["click_time_ms"] - r["decision_split_ms"]
        ax.barh(row_i, thinking_dur, left=0, color=colors["thinking"], label="thinking" if row_i == 0 else None)
        ax.barh(row_i, decision_dur, left=thinking_dur, color=colors["decision"],
                 label="decision" if row_i == 0 else None)
        ax.barh(row_i, DEFAULT_BUFFER_MS, left=thinking_dur + decision_dur, color=colors["motor"],
                 label="motor" if row_i == 0 else None)
        if r["is_estimated_click"]:
            ax.text(thinking_dur + decision_dur + DEFAULT_BUFFER_MS + 5, row_i, "(est. click, miss)",
                     va="center", fontsize=7, color="gray")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["trial_idx"].tolist())
    ax.set_xlabel("Time since stimulus onset (ms)")
    ax.set_ylabel("Trial")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out_path = f"{out_dir}/phase_timeline_{_exp_tag(exp_id)}_{kind}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ============================================================
# STEP 3: Interactive experiment selection
# ============================================================

def prompt_for_experiment(ids: list, kind: str) -> int:
    task_label = "attention" if kind == "attention" else "cognitive load"
    print(f"\nAvailable {task_label} experiment IDs:")
    print(", ".join(str(i) for i in ids))
    while True:
        choice = input(f"Pick an experiment number for {task_label}: ").strip()
        try:
            choice_int = int(choice)
            if choice_int in ids:
                return choice_int
            print(f"  {choice_int} is not in the list above -- try again.")
        except ValueError:
            print("  Please enter a valid integer.")


# ============================================================
# STEP 4: Per-experiment plots (output, match/detection, Pearson)
# No in-figure titles -- filenames encode what each plot shows.
# ============================================================

def _exp_tag(exp_id) -> str:
    """Filename tag: numeric ids -> "experiment_190"; strings (e.g. the
    pooled analysis's "all_experiments") are used as-is."""
    return f"experiment_{exp_id}" if str(exp_id).isdigit() else str(exp_id)


def plot_output_single_experiment(detect_success: list, exp_id, kind: str, out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(detect_success)), detect_success)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_xlabel("Event ID")
    ax.set_ylabel("Detect Success")
    out_path = f"{out_dir}/output_{_exp_tag(exp_id)}_{kind}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_hit_miss_single_experiment(detect_success: list, exp_id, kind: str, out_dir: str):
    """Hit/miss view for THIS experiment alone (the same experiment chosen
    for the Pearson correlation plots) -- distinct from
    plot_output_single_experiment()'s raw 0/1 line: here each trial is a
    colored marker (green = hit / detectSuccess=1, red = miss /
    detectSuccess=0) by trial (Event ID), plus a hit-vs-miss COUNT bar so
    the overall hit rate for this one experiment is visible at a glance."""
    detect_success = np.asarray(detect_success)
    block_ids = np.arange(len(detect_success))
    colors = ["#2ca02c" if v == 1 else "#d62728" for v in detect_success]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [3, 1]})

    axes[0].scatter(block_ids, detect_success, c=colors, s=60, zorder=3)
    axes[0].plot(block_ids, detect_success, color="gray", alpha=0.4, zorder=1)
    axes[0].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(["Miss", "Hit"])
    axes[0].set_xlabel("Event ID")
    axes[0].set_ylabel("Outcome")

    n_hit = int((detect_success == 1).sum())
    n_miss = int((detect_success == 0).sum())
    axes[1].bar(["Hit", "Miss"], [n_hit, n_miss], color=["#2ca02c", "#d62728"])
    axes[1].set_ylabel("Trial count")
    for i, v in enumerate([n_hit, n_miss]):
        axes[1].text(i, v, str(v), ha="center", va="bottom")

    plt.tight_layout()
    out_path = f"{out_dir}/hit_miss_{_exp_tag(exp_id)}_{kind}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_match_detection_vs_responsive_attention(pb_full: pd.DataFrame, match_times: list,
                                                     detect_success: list, exp_id, kind: str, out_dir: str):
    """Reproduces the original comparing_plots()/plot_each_powerband()
    style dual-axis figure: Detection success (blue, per-trial) vs.
    Responsive Attention = beta_f7 / (alpha_f7 + theta_f7) (red,
    continuous, full sliding-window resolution -- NOT block-averaged),
    with match_times rescaled onto the power-band array's own index
    range so the two, differently-sampled series can share one x-axis."""
    watt = pb_full["beta_f7"] / (pb_full["alpha_f7"] + pb_full["theta_f7"])

    coff = np.max(match_times) / len(watt)
    z = np.divide(match_times, coff)

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(z, detect_success, "b-", marker="o", label="Detection success")
    ax1.set_xlabel("Match times")
    ax1.set_ylabel("Detection success", color="b")
    ax1.tick_params("y", colors="b")

    ax2 = ax1.twinx()
    ax2.plot(watt.values, "r-", label="Responsive Attention")
    ax2.set_ylabel("Responsive Attention", color="r")
    ax2.tick_params("y", colors="r")

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right")

    out_path = f"{out_dir}/match_detection_vs_responsive_attention_{_exp_tag(exp_id)}_{kind}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_match_detection_vs_cognitive_load(pb_full: pd.DataFrame, match_times: list,
                                               detect_success: list, exp_id, kind: str, out_dir: str):
    """Second of the three original plot_each_powerband() figures:
    Detection success vs. Cognitive Load = (theta_f3 + theta_f4) /
    (alpha_p7 + alpha_p8)."""
    cl = (pb_full["theta_f3"] + pb_full["theta_f4"]) / (pb_full["alpha_p7"] + pb_full["alpha_p8"])

    coff = np.max(match_times) / len(cl)
    z = np.divide(match_times, coff)

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(z, detect_success, "b-", marker="o", label="Detection success")
    ax1.set_xlabel("Match times")
    ax1.set_ylabel("Detection success", color="b")
    ax1.tick_params("y", colors="b")

    ax2 = ax1.twinx()
    ax2.plot(cl.values, "r-", label="Cognitive Load")
    ax2.set_ylabel("Cognitive Load", color="r")
    ax2.tick_params("y", colors="r")

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right")

    out_path = f"{out_dir}/match_detection_vs_cognitive_load_{_exp_tag(exp_id)}_{kind}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_match_detection_vs_attention(pb_full: pd.DataFrame, match_times: list,
                                          detect_success: list, exp_id, kind: str, out_dir: str):
    """Third of the three original plot_each_powerband() figures:
    Detection success vs. Attention = beta_p7 / (alpha_p7 + theta_p7)."""
    eratt = pb_full["beta_p7"] / (pb_full["alpha_p7"] + pb_full["theta_p7"])

    coff = np.max(match_times) / len(eratt)
    z = np.divide(match_times, coff)

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(z, detect_success, "b-", marker="o", label="Detection success")
    ax1.set_xlabel("Match times")
    ax1.set_ylabel("Detection success", color="b")
    ax1.tick_params("y", colors="b")

    ax2 = ax1.twinx()
    ax2.plot(eratt.values, "r-", label="Attention")
    ax2.set_ylabel("Attention", color="r")
    ax2.tick_params("y", colors="r")

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right")

    out_path = f"{out_dir}/match_detection_vs_attention_{_exp_tag(exp_id)}_{kind}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_sensor_correlation_heatmap(trial_features: pd.DataFrame, exp_id, kind: str, out_dir: str):
    """Reproduces sensor_correlation_plot(): the FULL correlation
    matrix between every pair of sensor features for this one
    experiment (not just each sensor vs. outcome, which is the
    separate Pearson bar chart)."""
    correlation_matrix = trial_features.corr()
    fig = plt.figure(figsize=(20, 16))
    sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", fmt=".2f")
    out_path = f"{out_dir}/sensor_correlation_heatmap_{_exp_tag(exp_id)}_{kind}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


# Band-major sensor ordering/coloring, matching the layout used by the
# original CorrelationAnalysis.py / correlation_after_cleaning.py aggregate
# plots: all electrodes for "alpha" first, then "beta", "delta", "gamma",
# "theta", each band sharing one color so the five groups are visually
# distinguishable at a glance.
BAND_PLOT_ORDER = ["alpha", "beta", "delta", "gamma", "theta"]


def canonical_sensor_order(columns) -> list:
    """Reorders a list of '{band}_{electrode}' column names band-major
    (BAND_PLOT_ORDER x BCI_ELECTRODES), falling back to original order for
    anything that doesn't match that naming pattern."""
    columns = list(columns)
    ordered = [f"{band}_{e}" for band in BAND_PLOT_ORDER for e in BCI_ELECTRODES if f"{band}_{e}" in columns]
    leftover = [c for c in columns if c not in ordered]
    return ordered + leftover


def sensor_band_colors(sensor_names) -> list:
    """One color per band (matplotlib's default color cycle), repeated for
    every electrode within that band -- reproduces the banded coloring in
    the original aggregate correlation plots."""
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    band_color = {band: cycle[i % len(cycle)] for i, band in enumerate(BAND_PLOT_ORDER)}
    return [band_color.get(str(s).split("_")[0], "#888888") for s in sensor_names]


def compute_pearson_table_single_experiment(trial_features: pd.DataFrame, detect_success: np.ndarray,
                                                exp_id, kind: str, out_dir: str) -> pd.DataFrame:
    """Computes Pearson correlation AND p-value between each feature and
    outcome, for THIS experiment alone, and saves the result as a CSV
    table (sensor, correlation, p_value). This is the single source of
    truth used by both the correlation-value plot and the separate
    p-value plot below, so the table and the two plots can never drift
    out of sync with each other. Sensors are reordered band-major
    (canonical_sensor_order) so the table and both plots group all-alpha,
    all-beta, etc. together, matching the aggregate correlation analysis's
    layout instead of whatever column order the feature table happened
    to come in."""
    ordered_cols = canonical_sensor_order(trial_features.columns)
    trial_features = trial_features[ordered_cols]

    rows = []
    for col in trial_features.columns:
        try:
            corr, p_value = stats.pearsonr(trial_features[col], detect_success)
        except Exception:
            corr, p_value = np.nan, np.nan
        rows.append({"sensor": col, "correlation": corr, "p_value": p_value})

    table = pd.DataFrame(rows)
    out_path = f"{out_dir}/pearson_correlation_table_{_exp_tag(exp_id)}_{kind}.csv"
    table.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")
    return table


def plot_pearson_single_experiment(pearson_table: pd.DataFrame, exp_id, kind: str, out_dir: str):
    """Pearson correlation VALUE for each feature vs. outcome, computed
    for THIS experiment alone (not aggregated across experiments). Bars
    are grouped and colored by band (see canonical_sensor_order /
    sensor_band_colors), and the title is restored to match the exact
    style of the original source this plot is modeled on: CorrelationAnalysis.py's
    plot_correlation_results(), which titles its figure
    f'Correlation Results ({{method.capitalize()}})' -- here hardcoded to
    "Correlation Results (Pearson)" since this function only ever computes
    Pearson."""
    colors = sensor_band_colors(pearson_table["sensor"])
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(range(len(pearson_table)), pearson_table["correlation"], color=colors)
    ax.set_xticks(range(len(pearson_table)))
    ax.set_xticklabels(pearson_table["sensor"], rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Sensors")
    ax.set_ylabel("Correlation (Pearson)")
    plt.tight_layout()
    out_path = f"{out_dir}/pearson_correlation_{_exp_tag(exp_id)}_{kind}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_pearson_pvalues_single_experiment(pearson_table: pd.DataFrame, exp_id, kind: str, out_dir: str):
    """P-VALUE for each feature's Pearson correlation with outcome, for
    THIS experiment alone. Kept as its own separate figure -- never
    combined with the correlation-value plot above -- matching the
    convention already used for the aggregate correlation analysis
    (correlation_after_cleaning_v2.py's plot_correlation_values() /
    plot_p_values()). A horizontal reference line at p=0.05 is drawn
    for quick visual reference; it is NOT a multiple-comparisons
    correction (70 sensors are tested here, so a naive 0.05 cutoff
    will produce some false positives by chance -- see paper caveats).
    Same band grouping/coloring as plot_pearson_single_experiment().
    Titled -- note the ORIGINAL CorrelationAnalysis.plot_p_value() copy-
    pasted the correlation plot's title verbatim ('Correlation Results
    (Pearson)') onto the p-value figure too, so the two were
    indistinguishable by title alone; fixed here with its own title."""
    colors = sensor_band_colors(pearson_table["sensor"])
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(range(len(pearson_table)), pearson_table["p_value"], color=colors)
    ax.axhline(0.05, color="red", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(pearson_table)))
    ax.set_xticklabels(pearson_table["sensor"], rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Sensors")
    ax.set_ylabel("p-value (Pearson)")
    plt.tight_layout()
    out_path = f"{out_dir}/pearson_pvalues_{_exp_tag(exp_id)}_{kind}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def filter_only(raw):
    """Same band-pass + average reference as clean_raw(), WITHOUT ICA."""
    raw_f = raw.copy().filter(l_freq=1.0, h_freq=45.0, fir_design="firwin")
    raw_f.set_eeg_reference("average")
    return raw_f


def generate_single_experiment_plots(exp_id: int, kind: str, X_clean: pd.DataFrame,
                                         y: np.ndarray, exp_ids: np.ndarray, use_ica: bool = True):
    task_label = "attention" if kind == "attention" else "cognitive load"
    out_dir = RESULTS_DIR[kind]
    os.makedirs(out_dir, exist_ok=True)
    pb_dir, eeg_dir = PB_DIR[kind], EEG_DIR[kind]

    label_path = f"{pb_dir}/{exp_id}-{kind}.json"
    eeg_path = f"{eeg_dir}/{exp_id}-{kind}-eeg.json"
    rec = load_labels(label_path)
    detect_success = rec["detectSuccess"]
    match_times = rec["match_times"]

    plot_output_single_experiment(detect_success, exp_id, kind, out_dir)
    plot_hit_miss_single_experiment(detect_success, exp_id, kind, out_dir)

    # Full-resolution (not block-averaged) CLEANED power bands, for the
    # three match/detection-vs-power-metric plots (Cognitive Load,
    # Attention, Responsive Attention -- all three from the original
    # plot_each_powerband())
    raw = load_eeg_export(eeg_path)
    pb_full = sliding_window_power_bands(clean_raw(raw)["raw_clean"] if use_ica else filter_only(raw))
    plot_match_detection_vs_cognitive_load(pb_full, match_times, detect_success, exp_id, kind, out_dir)
    plot_match_detection_vs_attention(pb_full, match_times, detect_success, exp_id, kind, out_dir)
    plot_match_detection_vs_responsive_attention(pb_full, match_times, detect_success, exp_id, kind, out_dir)

    # Per-trial CLEANED features for THIS experiment only, from the
    # already-built dataset (avoids recomputing)
    mask = exp_ids == exp_id
    if mask.sum() > 0:
        features_this_exp = X_clean[mask].reset_index(drop=True)
        features_this_exp = features_this_exp[canonical_sensor_order(features_this_exp.columns)]
        pearson_table = compute_pearson_table_single_experiment(features_this_exp, y[mask], exp_id, kind, out_dir)
        plot_pearson_single_experiment(pearson_table, exp_id, kind, out_dir)
        plot_pearson_pvalues_single_experiment(pearson_table, exp_id, kind, out_dir)
        plot_sensor_correlation_heatmap(features_this_exp, exp_id, kind, out_dir)
    else:
        print(f"[skip] Pearson/correlation plots for experiment {exp_id}: not found in the built dataset "
              f"(likely using a cached cleaned dataset from before this experiment was added).")


def run_pooled_correlation(X: pd.DataFrame, y: np.ndarray, kind: str, out_dir: str):
    """Pearson correlation of every power-band feature with hit/miss,
    pooled across ALL experiments of this task (table + correlation plot +
    separate p-value plot + feature-by-feature heatmap)."""
    os.makedirs(out_dir, exist_ok=True)
    features = X[canonical_sensor_order(X.columns)].reset_index(drop=True)
    table = compute_pearson_table_single_experiment(features, np.asarray(y), "all_experiments", kind, out_dir)
    plot_pearson_single_experiment(table, "all_experiments", kind, out_dir)
    plot_pearson_pvalues_single_experiment(table, "all_experiments", kind, out_dir)
    plot_sensor_correlation_heatmap(features, "all_experiments", kind, out_dir)


# ============================================================
# STEP 5: Aggregate ICA diagnostics (component exclusion, channel effect)
# ============================================================

def plot_component_exclusion_summary(ica_diagnostics: list, kind: str, out_dir: str):
    """Two SEPARATE figures (not subplots of one): (1) how often each
    artifact type was removed, with large x-axis labels (eye blink,
    muscle artifact, channel noise, heart beat, ...); (2) histogram of how
    many components were excluded per experiment."""
    all_labels = [l for d in ica_diagnostics for l in d["excluded_labels"]]
    label_counts = pd.Series(all_labels).value_counts()
    n_total = len(ica_diagnostics)

    # --- Figure 1: count per artifact type ---
    fig, ax = plt.subplots(figsize=(11, 6.5))
    if len(label_counts) > 0:
        ax.bar(label_counts.index, label_counts.values, color="steelblue")
    ax.set_ylabel(f"Count across {n_total} experiments", fontsize=18)
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=16)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    out_path = f"{out_dir}/component_exclusion_by_artifact_type_{kind}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")

    # --- Figure 2: components excluded per experiment ---
    n_excluded = [len(d["excluded_labels"]) for d in ica_diagnostics]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.hist(n_excluded, bins=range(0, max(n_excluded, default=0) + 2), color="steelblue",
             align="left", rwidth=0.8)
    ax.set_xlabel("Number of components excluded", fontsize=18)
    ax.set_ylabel("Number of experiments", fontsize=18)
    ax.tick_params(axis="both", labelsize=16)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    out_path = f"{out_dir}/component_exclusion_count_per_experiment_{kind}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_channel_effect_distribution(ica_diagnostics: list, kind: str, out_dir: str):
    channel_df = pd.DataFrame([{k: v for k, v in d.items() if k in EMOTIV_CHANNELS} for d in ica_diagnostics])
    channel_order = channel_df.median().sort_values().index

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot([channel_df[ch].dropna() for ch in channel_order], tick_labels=channel_order)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("% change in power (after vs. before ICA cleaning)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    out_path = f"{out_dir}/channel_effect_distribution_{kind}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


# ============================================================
# STEP 6: Models -- Logistic, Logistic+ANOVA, Linear Regression, and
# the user's chosen neural network
# ============================================================

def build_first_network(input_dim: int):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(input_dim * 2, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(input_dim, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(max(1, input_dim // 4), activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.01), metrics=["accuracy"])
    return model, {"epochs": 64, "batch_size": 16}


def build_second_network(input_dim: int):
    model = Sequential()
    model.add(Dense(units=32, activation="relu", input_dim=input_dim))
    model.add(Dense(units=16, activation="relu"))
    model.add(Dense(units=1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model, {"epochs": 64, "batch_size": 32}


def build_third_network(input_dim: int):
    model = Sequential()
    model.add(Dense(units=32, activation="relu", input_dim=input_dim))
    model.add(Dropout(0.4))
    model.add(Dense(units=16, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(units=8, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(units=1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model, {"epochs": 128, "batch_size": 32}


# All three architectures report under the plain name "Neural Network" so
# plots/CSVs never say "First/Second/Third Network" -- the architecture
# choice is still made at the prompt and printed there.
NETWORK_BUILDERS = {"1": ("Neural Network", build_first_network),
                     "2": ("Neural Network", build_second_network),
                     "3": ("Neural Network", build_third_network)}


ALL_MODELS = ["logistic", "logistic_anova", "linear", "nn", "hmm"]
MODEL_MENU = [("logistic", "Logistic Regression"),
              ("logistic_anova", "Logistic Regression + ANOVA feature selection"),
              ("linear", "Linear Regression (sigmoid + Youden threshold)"),
              ("nn", "Neural Network"),
              ("hmm", "Hidden Markov Model")]


def prompt_for_network_choice() -> tuple:
    print("\nChoose a neural network architecture:")
    print("  1. Architecture 1   (Dense(2n) -> Dropout -> Dense(n) -> Dropout -> Dense(n/4) -> Dense(1))")
    print("  2. Architecture 2   (Dense(64) -> Dense(32) -> Dense(1))")
    print("  3. Architecture 3   (Dense(128) -> Dropout -> Dense(64) -> Dropout -> Dense(16) -> Dense(1))")
    choice = ask("Enter 1, 2, or 3", ["1", "2", "3"], default="2")
    name, builder = NETWORK_BUILDERS[choice]
    print(f"Selected: architecture {choice}")
    return name, builder


def _scores(y_test, y_pred) -> dict:
    """accuracy + balanced accuracy, plus the raw predictions so the
    leave-one-subject-out driver can pool them across folds. Balanced
    accuracy is NaN when the test set has a single class (the held-out
    subject may have only hits or only misses)."""
    y_pred = np.asarray(y_pred).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bal = balanced_accuracy_score(y_test, y_pred) if len(np.unique(y_test)) > 1 else np.nan
    return {"accuracy": accuracy_score(y_test, y_pred), "balanced_accuracy": bal, "y_pred": y_pred}


def train_and_evaluate_nn(builder, X_train, X_test, y_train, y_test) -> dict:
    model, fit_kwargs = builder(X_train.shape[1])
    # No validation_data=(X_test, y_test) here: the held-out subject must
    # not be touched at all during training (it was only a monitor, never
    # used for selection, but LOSO should be unambiguous about this).
    model.fit(X_train, y_train, epochs=fit_kwargs["epochs"], batch_size=fit_kwargs["batch_size"], verbose=0)
    y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).ravel()
    return _scores(y_test, y_pred)


def train_and_evaluate_logistic(X_train, X_test, y_train, y_test) -> dict:
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return _scores(y_test, y_pred)


def train_and_evaluate_logistic_anova(X_train, X_test, y_train, y_test, k: int = 14) -> dict:
    k = min(k, X_train.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_sel, y_train)
    y_pred = model.predict(X_test_sel)
    return _scores(y_test, y_pred)


def train_and_evaluate_linear(X_train, X_test, y_train, y_test) -> dict:
    """Linear regression used as a binary classifier: fit continuous
    targets (0/1), pass predictions through a sigmoid, threshold using
    Youden's J statistic computed on the TRAINING predictions (never
    the test set)."""
    model = LinearRegression()
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    train_prob = expit(train_pred)  # sigmoid
    fpr, tpr, thresholds = roc_curve(y_train, train_prob)
    threshold = thresholds[np.argmax(tpr - fpr)]

    test_pred = model.predict(X_test)
    test_prob = expit(test_pred)
    y_pred = (test_prob > threshold).astype(int)
    return _scores(y_test, y_pred)


def train_and_evaluate_hmm(X_train, X_test, y_train, y_test) -> dict:
    """Hidden Markov Model classifier, reproducing ModelTraininer.py's
    HMM_regression(): a 2-state GaussianHMM fit on the (scaled) feature
    sequence, used to assign each trial to one of two latent states.

    Fix applied relative to the original code: HMM.fit()/.predict() is
    UNSUPERVISED -- the model only learns two latent states and has no
    idea which state number ("0" or "1") corresponds to which class
    label. The original code called `model.predict(X_test)` and compared
    the resulting state sequence directly to `y_test`, which silently
    assumes state 0 == class 0 and state 1 == class 1; with no such
    guarantee, accuracy could come out inverted (e.g. a model with 90%
    real agreement could be scored at 10%) depending on arbitrary EM
    initialization. Here, the state-to-label mapping is learned by
    majority vote on the TRAINING labels only (never on test), then
    applied to the predicted test states, which is the standard fix for
    this "label switching" problem in unsupervised sequence models.

    Made resilient on top of that: with very few trials (e.g. one
    experiment's own ~20-30 rows, as compute_accuracy_across_experiments
    uses), the EM fit can converge to a degenerate transition matrix --
    hmmlearn then raises on model.predict() ('transmat_ rows must sum to
    1'/zero-sum rows) rather than returning a bad-but-usable prediction.
    Every failure mode here is caught and turned into NaN + a 'skipped'
    reason instead of propagating, so a single flaky experiment can't
    desync a caller's per-model result lists (that exact crash happened
    before this fix: log/linear accuracy got appended for an experiment
    before HMM raised, leaving the four per-model lists different
    lengths when later zipped into a DataFrame)."""
    if not HMMLEARN_AVAILABLE:
        return {"accuracy": np.nan, "balanced_accuracy": np.nan, "skipped": "hmmlearn not installed"}

    try:
        model = hmm.GaussianHMM(n_components=2, covariance_type="diag", random_state=42)
        model.fit(X_train)

        train_states = model.predict(X_train)
        # Majority-vote state -> label mapping, learned on TRAIN only.
        state_to_label = {}
        for state in (0, 1):
            mask = train_states == state
            if mask.sum() == 0:
                state_to_label[state] = 0
                continue
            values, counts = np.unique(y_train[mask], return_counts=True)
            state_to_label[state] = values[np.argmax(counts)]

        test_states = model.predict(X_test)
        y_pred = np.array([state_to_label[s] for s in test_states])
    except Exception as e:
        return {"accuracy": np.nan, "balanced_accuracy": np.nan, "skipped": f"HMM fit/predict failed: {e}"}

    out = _scores(y_test, y_pred)
    out["transmat"] = model.transmat_.tolist()
    return out


def loso_predictions(X, y, groups, nn_name: str, nn_builder, models=None) -> dict:
    """Leave-One-Subject-Out cross-validation (reviewer comment 4).

    Every distinct value in `groups` (one per SUBJECT) is held out once:
    all of that subject's trials are the test set, and every model is
    trained from scratch on ALL the other subjects' trials. The
    StandardScaler, ANOVA feature selector, linear-regression threshold
    (Youden's J) and HMM state->label mapping are all fit on the training
    subjects only, and the neural network is rebuilt fresh per fold, so
    nothing about the held-out subject reaches training -- unlike the old
    shuffled 80/20 split, where one subject's trials sat on both sides of
    the split.

    Returns {model_name: {"y_pred": int array aligned to X's rows,
    "evaluated": bool mask of rows that actually got a prediction}}. A
    row is unevaluated if its fold was skipped (training subjects had only
    one outcome class) or the model failed on that fold (HMM can
    degenerate) -- failures never desync the other models."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    subjects = sorted(set(groups.tolist()), key=str)
    if len(subjects) < 2:
        return {}

    selected = set(models) if models else set(ALL_MODELS)
    model_fns = {}
    if "logistic" in selected:
        model_fns["Logistic Regression"] = train_and_evaluate_logistic
    if "logistic_anova" in selected:
        model_fns["Logistic Regression + ANOVA"] = train_and_evaluate_logistic_anova
    if "linear" in selected:
        model_fns["Linear Regression"] = train_and_evaluate_linear
    if "nn" in selected:
        if KERAS_AVAILABLE:
            model_fns[nn_name] = lambda a, b, c, d: train_and_evaluate_nn(nn_builder, a, b, c, d)
        else:
            print(f"  [skip] {nn_name}: tensorflow/keras not installed")
    if "hmm" in selected:
        if HMMLEARN_AVAILABLE:
            model_fns["HMM"] = train_and_evaluate_hmm
        else:
            print("  [skip] HMM: hmmlearn not installed")

    preds = {m: np.full(len(y), -1, dtype=int) for m in model_fns}
    n_skipped_folds = 0
    for subject in subjects:
        test_mask = groups == subject
        train_mask = ~test_mask
        if len(np.unique(y[train_mask])) < 2:
            n_skipped_folds += 1
            continue
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_mask])
        X_test = scaler.transform(X[test_mask])
        y_train, y_test = y[train_mask], y[test_mask]
        for name, fn in model_fns.items():
            try:
                out = fn(X_train, X_test, y_train, y_test)
                if "y_pred" in out:
                    preds[name][test_mask] = out["y_pred"]
            except Exception as e:
                print(f"  [error] {name} (held-out subject {subject}): {e}")
        if KERAS_AVAILABLE:
            try:
                from tensorflow.keras import backend as K
                K.clear_session()  # one fresh NN per fold; don't accumulate graphs
            except Exception:
                pass
    if n_skipped_folds:
        print(f"  [note] {n_skipped_folds} held-out subject(s) skipped: training subjects had one outcome class only")
    return {m: {"y_pred": p, "evaluated": p >= 0} for m, p in preds.items()}


def run_models_on_version(X: pd.DataFrame, y: np.ndarray, nn_name: str, nn_builder, groups, models=None) -> dict:
    """LOSO evaluation of all models on one feature set. `groups` = the
    subject of each row. Metrics are computed on the POOLED out-of-subject
    predictions (every row predicted exactly once, by a model that never
    saw that row's subject). `mean_subject_accuracy` (unweighted mean of
    each held-out subject's own accuracy) and `n_subjects` are also
    reported, since subjects have different trial counts."""
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    loso = loso_predictions(X, y, groups, nn_name, nn_builder, models)
    results = {}
    if not loso:
        print("  [skip] LOSO needs at least 2 subjects.")
        return results
    for name, d in loso.items():
        mask = d["evaluated"]
        if mask.sum() == 0:
            results[name] = {"accuracy": np.nan, "balanced_accuracy": np.nan,
                              "mean_subject_accuracy": np.nan, "n_subjects": 0}
            continue
        yt, yp = y[mask], d["y_pred"][mask]
        per_subject = [accuracy_score(yt[groups[mask] == g], yp[groups[mask] == g])
                       for g in sorted(set(groups[mask].tolist()), key=str)]
        results[name] = {"accuracy": accuracy_score(yt, yp),
                          "balanced_accuracy": balanced_accuracy_score(yt, yp) if len(np.unique(yt)) > 1 else np.nan,
                          "mean_subject_accuracy": float(np.mean(per_subject)),
                          "n_subjects": len(per_subject)}
    return results


def compute_accuracy_across_experiments(X_clean: pd.DataFrame, y: np.ndarray, exp_ids: np.ndarray,
                                            subjects: np.ndarray, nn_name: str, nn_builder, models=None) -> dict:
    """Figure 6 under LOSO: one accuracy per model per EXPERIMENT. Each
    experiment's value is the accuracy on that experiment's trials when
    its SUBJECT was held out -- the model that scored it was trained only
    on other subjects' experiments (the old version split 80/20 inside
    each experiment, ~20-30 trials). Experiments with fewer than 6 trials,
    a single outcome class, or no model prediction are skipped, as before."""
    y = np.asarray(y).astype(int)
    loso = loso_predictions(X_clean, y, subjects, nn_name, nn_builder, models)
    out = {"exp_ids": [], "nn_accuracy": [], "logistic_accuracy": [], "linear_accuracy": [],
           "hmm_accuracy": [], "nn_name": nn_name}
    if not loso:
        return out
    key = {"nn_accuracy": nn_name, "logistic_accuracy": "Logistic Regression",
           "linear_accuracy": "Linear Regression", "hmm_accuracy": "HMM"}
    for exp_id in sorted(set(exp_ids.tolist())):
        mask = exp_ids == exp_id
        if len(np.unique(y[mask])) < 2 or mask.sum() < 6:
            continue
        for field, model_name in key.items():
            d = loso.get(model_name)
            if d is None:
                out[field].append(np.nan)
                continue
            m = mask & d["evaluated"]
            out[field].append(accuracy_score(y[m], d["y_pred"][m]) if m.sum() else np.nan)
        out["exp_ids"].append(exp_id)
    return out



def plot_accuracy_across_experiments(result: dict, kind: str, out_dir: str):
    """Figure 6: reproduces PsyPlot.py's plotAccuracyExperiments(), now
    with all FOUR models (NN, Logistic, Linear, HMM) in one plot. Each
    "block" here is one EXPERIMENT (one recording session): block_numbers
    is just the 0..N-1 position of that experiment in the sorted list of
    experiment IDs used for this run (NOT a trial index and NOT the raw
    experiment ID itself), matching the original plotAccuracyExperiments()
    semantics of one accuracy point per experiment. Forced to integer
    ticks below -- with only a handful of experiments, matplotlib's
    default locator was labeling ticks 7.5, 10.5, etc., which isn't a
    meaningful block number."""
    block_numbers = list(range(len(result["exp_ids"])))
    fig, ax = plt.subplots(figsize=(10, 6))
    series = [("nn_accuracy", f"{result['nn_name']} accuracy"),
               ("logistic_accuracy", "Logistic regression accuracy"),
               ("hmm_accuracy", "HMM accuracy"),
               ("linear_accuracy", "Linear regression accuracy")]
    for field, label in series:
        values = result[field]
        if len(values) and not np.all(np.isnan(np.asarray(values, dtype=float))):
            ax.plot(block_numbers, values, marker="o", label=label)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_xlabel("Block Number")
    ax.set_ylabel("Accuracy")
    ax.legend()
    out_path = f"{out_dir}/accuracy_across_experiments_{kind}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_clean_vs_uncleaned(model_results: dict, kind: str, out_dir: str):
    model_names = list(model_results.keys())
    fig, ax = plt.subplots(figsize=(max(9, len(model_names) * 2.2), 5.5))
    x = np.arange(len(model_names))
    width = 0.35

    cleaned_acc = [model_results[m]["cleaned"]["balanced_accuracy"] for m in model_names]
    uncleaned_acc = [model_results[m]["uncleaned"]["balanced_accuracy"] for m in model_names]

    ax.bar(x - width/2, uncleaned_acc, width, label="Uncleaned", color="#d9776b")
    ax.bar(x + width/2, cleaned_acc, width, label="Cleaned", color="#5b9bd5")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylabel("Balanced accuracy")
    ax.legend()
    plt.tight_layout()
    out_path = f"{out_dir}/clean_vs_uncleaned_by_model_{kind}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


# ============================================================
# STEP 7: Save cleaned data (also serves as Step 2's cache for next run)
# ============================================================

def save_clean_data(X_clean: pd.DataFrame, y: np.ndarray, subjects: np.ndarray, exp_ids: np.ndarray, kind: str):
    out_dir = CLEAN_DATA_DIR[kind]
    os.makedirs(out_dir, exist_ok=True)
    df = X_clean.copy()
    df["Sucess"] = y
    df["subject"] = subjects
    df["exp_id"] = exp_ids
    out_path = f"{out_dir}/cleaned_power_bands.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved cleaned data -> {out_path}  ({len(df)} rows)")


# ============================================================
# MAIN
# ============================================================

def build_uncleaned_dataset_only(ids: list, kind: str) -> dict:
    """Dataset used when the user skips ICA: trial-averaged power bands of
    the raw (unfiltered-by-ICA) signal, in the same dict shape as
    build_cleaned_and_uncleaned()."""
    print(f"\n{'#'*70}\nSTEP 2: ICA cleaning SKIPPED -- using the uncleaned power bands\n{'#'*70}")
    data = build_uncleaned_only(ids, kind)
    return {"cleaned": data, "uncleaned": data, "ica_diagnostics": None, "no_ica": True}


def run_for_kind(ids: list, kind: str, opts: dict):
    """Runs the selected stages for ONE task ("attention" or "memory").
    opts: use_ica (bool), correlation ("none"|"single"|"pooled"|"both"),
    models (list of keys from ALL_MODELS), nn_name, nn_builder,
    phases (bool: also run the thinking-window / phase analyses),
    experiment (optional int, for the single-experiment plots)."""
    task_label = "attention" if kind == "attention" else "cognitive load / memory"
    results_dir = RESULTS_DIR[kind]
    os.makedirs(results_dir, exist_ok=True)
    use_ica, models = opts["use_ica"], list(opts["models"])
    nn_name, nn_builder = opts["nn_name"], opts["nn_builder"]

    print(f"\n{'='*70}\n{task_label.upper()}\n{'='*70}")

    # --- Step 2: ICA cleaning (or skip) -- every later stage reads from this ---
    datasets = build_cleaned_and_uncleaned(ids, kind) if use_ica else build_uncleaned_dataset_only(ids, kind)
    X_clean, y, subjects, exp_ids = datasets["cleaned"]
    X_uncleaned, y_uncleaned, subjects_uncleaned, _ = datasets["uncleaned"]
    # Normally the uncleaned dataset is row-for-row the same trials as the
    # cleaned one. The exception is a cache built from a different
    # experiment set than what is in the data folder now -- then the
    # uncleaned dataset's own labels/subjects are used for its models.
    y_for_uncleaned = y_uncleaned if datasets.get("uncleaned_y_mismatch") else y
    groups_for_uncleaned = subjects_uncleaned if datasets.get("uncleaned_y_mismatch") else subjects

    if len(X_clean) == 0:
        print(f"No usable data for {task_label}.")
        return {}

    # exp_id -> participant label, so the thinking-window / phase datasets
    # (which only carry experiment ids) can be grouped for leave-one-subject-out.
    exp_to_subject = dict(zip(exp_ids.tolist(), subjects.tolist()))

    def _subjects_for(exp_id_array):
        return np.array([exp_to_subject.get(e, f"exp{e}") for e in np.asarray(exp_id_array).tolist()])

    # --- Correlation ---
    chosen_exp = None
    if opts["correlation"] in ("single", "both"):
        available = sorted(set(exp_ids.tolist()))
        chosen_exp = opts.get("experiment")
        if chosen_exp not in available:
            chosen_exp = prompt_for_experiment(available, kind) if opts["interactive"] else available[0]
        generate_single_experiment_plots(chosen_exp, kind, X_clean, y, exp_ids, use_ica)
    if opts["correlation"] in ("pooled", "both"):
        print(f"\nPooled Pearson correlation across all {kind} experiments...")
        run_pooled_correlation(X_clean, y, kind, results_dir)

    # --- Aggregate ICA diagnostics ---
    if use_ica:
        if datasets["ica_diagnostics"]:
            plot_component_exclusion_summary(datasets["ica_diagnostics"], kind, results_dir)
            plot_channel_effect_distribution(datasets["ica_diagnostics"], kind, results_dir)
        else:
            print(f"\n[skip] ICA diagnostic plots: the cleaned data was loaded from its cache and no "
                  f"saved ICA diagnostics exist. Delete {CLEAN_DATA_DIR[kind]}/ and re-run to regenerate them.")

    results_main = {}
    if models:
        # --- Models: leave-one-subject-out ---
        print(f"\n{'#'*70}\nMODEL TRAINING (leave-one-subject-out) -- {task_label.upper()}\n{'#'*70}")
        if use_ica:
            print("Running on CLEANED data...")
        results_cleaned = run_models_on_version(X_clean, y, nn_name, nn_builder, subjects, models)
        results_main = results_cleaned

        if use_ica:
            print("Running on UNCLEANED data...")
            results_uncleaned = run_models_on_version(X_uncleaned, y_for_uncleaned, nn_name, nn_builder,
                                                       groups_for_uncleaned, models)
            combined = {}
            for model_name in results_cleaned:
                combined[model_name] = {"cleaned": results_cleaned[model_name],
                                          "uncleaned": results_uncleaned.get(
                                              model_name, {"accuracy": np.nan, "balanced_accuracy": np.nan})}
                print(f"  {model_name}: cleaned balanced_acc={results_cleaned[model_name]['balanced_accuracy']:.3f}, "
                      f"uncleaned balanced_acc={combined[model_name]['uncleaned']['balanced_accuracy']:.3f}")
            rows = [{"model": m, "version": v, "accuracy": r.get("accuracy", np.nan),
                      "balanced_accuracy": r.get("balanced_accuracy", np.nan)}
                     for m, versions in combined.items() for v, r in versions.items()]
            pd.DataFrame(rows).to_csv(f"{results_dir}/clean_vs_uncleaned_by_model_{kind}.csv", index=False)
            print(f"Saved -> {results_dir}/clean_vs_uncleaned_by_model_{kind}.csv")
            plot_clean_vs_uncleaned(combined, kind, results_dir)
        else:
            rows = [{"model": m, "accuracy": r.get("accuracy", np.nan),
                      "balanced_accuracy": r.get("balanced_accuracy", np.nan)} for m, r in results_cleaned.items()]
            for r in rows:
                print(f"  {r['model']}: accuracy={r['accuracy']:.3f}, balanced_acc={r['balanced_accuracy']:.3f}")
            pd.DataFrame(rows).to_csv(f"{results_dir}/model_accuracy_uncleaned_{kind}.csv", index=False)
            print(f"Saved -> {results_dir}/model_accuracy_uncleaned_{kind}.csv")

        # --- Accuracy across experiments (Figure 6) ---
        print(f"\nComputing per-experiment accuracy -- {task_label.upper()}...")
        accuracy_result = compute_accuracy_across_experiments(X_clean, y, exp_ids, subjects, nn_name, nn_builder, models)
        if accuracy_result["exp_ids"]:
            n_total_experiments = len(set(exp_ids.tolist()))
            print(f"  {len(accuracy_result['exp_ids'])}/{n_total_experiments} {kind} experiments used (the rest "
                  f"had fewer than 6 trials, or only one outcome class, and were skipped; each value = that "
                  f"experiment scored by a model trained WITHOUT its participant)")
            cols = {"exp_id": accuracy_result["exp_ids"]}
            for field, col in (("nn_accuracy", f"{nn_name}_accuracy"), ("logistic_accuracy", "logistic_accuracy"),
                                ("linear_accuracy", "linear_accuracy"), ("hmm_accuracy", "hmm_accuracy")):
                if not np.all(np.isnan(np.asarray(accuracy_result[field], dtype=float))):
                    cols[col] = accuracy_result[field]
            pd.DataFrame(cols).to_csv(f"{results_dir}/accuracy_across_experiments_{kind}.csv", index=False)
            print(f"Saved -> {results_dir}/accuracy_across_experiments_{kind}.csv")
            plot_accuracy_across_experiments(accuracy_result, kind, results_dir)
        else:
            print("[skip] per-experiment accuracy: no experiments had enough data.")

        # --- Thinking-window and thinking/decision/motor phase analyses ---
        if opts["phases"] and not use_ica:
            print("\n[skip] Thinking-window / phase analyses need the ICA-cleaned signal -- run with ICA cleaning enabled.")
        elif opts["phases"]:
            print(f"\n{'#'*70}\nTHINKING-PERIOD-ONLY MODELS -- {task_label.upper()}\n{'#'*70}")
            X_thinking, y_thinking, exp_ids_thinking, is_est_thinking = build_thinking_only_dataset_for_kind(ids, kind)
            if len(X_thinking) > 0:
                print(f"  {len(X_thinking)} trials ({int(is_est_thinking.sum())} with an estimated click time for a miss)")
                thinking_results = run_models_on_version(X_thinking, y_thinking, nn_name, nn_builder,
                                                          _subjects_for(exp_ids_thinking), models)
                for model_name, metrics in thinking_results.items():
                    print(f"  {model_name}: thinking-only accuracy={metrics.get('accuracy', np.nan):.3f} "
                          f"(whole-trial was {results_cleaned.get(model_name, {}).get('accuracy', np.nan):.3f})")
                pd.DataFrame([{"model": m, "whole_trial_accuracy": results_cleaned.get(m, {}).get("accuracy", np.nan),
                                "thinking_only_accuracy": r.get("accuracy", np.nan),
                                "thinking_only_balanced_accuracy": r.get("balanced_accuracy", np.nan)}
                               for m, r in thinking_results.items()]
                              ).to_csv(f"{results_dir}/thinking_period_models_{kind}.csv", index=False)
                print(f"Saved -> {results_dir}/thinking_period_models_{kind}.csv")
            else:
                print("  [skip] no usable trials (needs the cleaned power bands written by the ICA step).")

            print(f"\n{'#'*70}\nTHINKING / DECISION / MOTOR PHASES -- {task_label.upper()}\n{'#'*70}")
            phase_data = build_phase_averaged_dataset_for_kind(ids, kind)
            if len(phase_data["trial_boundaries"]) > 0:
                trial_boundaries = phase_data["trial_boundaries"]
                trial_boundaries.to_csv(f"{results_dir}/phase_boundaries_{kind}.csv", index=False)
                print(f"Saved -> {results_dir}/phase_boundaries_{kind}.csv  "
                      f"({len(trial_boundaries)} trials; mean decision split at "
                      f"{trial_boundaries['decision_split_frac'].mean():.1%} of reaction time)")
                timeline_exp = chosen_exp if chosen_exp in set(trial_boundaries["exp_id"].tolist()) \
                    else int(trial_boundaries["exp_id"].iloc[0])
                plot_phase_timeline_single_experiment(trial_boundaries, timeline_exp, kind, results_dir)

                phase_accuracy_rows = []
                for phase_name in ("thinking", "decision", "motor"):
                    X_phase, y_phase = phase_data[phase_name]
                    if len(X_phase) == 0:
                        continue
                    phase_results = run_models_on_version(X_phase, y_phase, nn_name, nn_builder,
                                                           _subjects_for(phase_data["exp_ids"]), models)
                    for model_name, metrics in phase_results.items():
                        phase_accuracy_rows.append({"model": model_name, "phase": phase_name,
                                                      "accuracy": metrics.get("accuracy", np.nan),
                                                      "balanced_accuracy": metrics.get("balanced_accuracy", np.nan)})
                if phase_accuracy_rows:
                    phase_acc_df = pd.DataFrame(phase_accuracy_rows)
                    phase_acc_df.to_csv(f"{results_dir}/phase_model_accuracy_{kind}.csv", index=False)
                    print(f"Saved -> {results_dir}/phase_model_accuracy_{kind}.csv")
                    print(phase_acc_df.pivot(index="model", columns="phase", values="accuracy").to_string())
            else:
                print("  [skip] no usable hit/miss trials with a cached cleaned EEG signal.")

    # --- Save the cleaned data (also the next run's cache). Never written when ICA was skipped. ---
    if use_ica:
        save_clean_data(X_clean, y, subjects, exp_ids, kind)

    return {model_name: metrics["accuracy"] for model_name, metrics in results_main.items()}


MODEL_SUMMARY_DISPLAY_NAMES = {
    "Logistic Regression": "Logistic regression",
    "Linear Regression": "Linear regression",
    "HMM": "Hidden Markov Model",
    # Neural network entries are keyed by whichever architecture name was
    # chosen at the prompt (e.g. "First Network") -- relabeled generically
    # here since the summary table reports one technique per row.
}


def build_model_summary_table(attention_accuracy: dict, memory_accuracy: dict, nn_name: str) -> pd.DataFrame:
    """Cross-task summary table: one row per ML technique, with its
    overall (cleaned-data) accuracy identifying the attention pattern vs.
    the cognitive-load pattern, side by side -- the table layout requested
    directly: 'ML technique | Accuracy of ident. of attention pattern |
    Accuracy of ident. of cognitive load pattern'."""
    rename = dict(MODEL_SUMMARY_DISPLAY_NAMES)
    rename[nn_name] = "Neural Network"

    all_models = list(dict.fromkeys(list(attention_accuracy.keys()) + list(memory_accuracy.keys())))
    rows = []
    for model_name in all_models:
        rows.append({
            "ML technique": rename.get(model_name, model_name),
            "Accuracy of ident. of attention pattern": attention_accuracy.get(model_name, np.nan),
            "Accuracy of ident. of cognitive load pattern": memory_accuracy.get(model_name, np.nan),
        })
    return pd.DataFrame(rows)


def plot_model_summary(summary_table: pd.DataFrame, out_dir: str):
    """All four models' accuracy for BOTH tasks in one grouped-bar plot --
    the table above, visualized."""
    models = summary_table["ML technique"]
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(9, len(models) * 2.2), 5.5))
    ax.bar(x - width/2, summary_table["Accuracy of ident. of attention pattern"],
            width, label="Attention pattern")
    ax.bar(x + width/2, summary_table["Accuracy of ident. of cognitive load pattern"],
            width, label="Cognitive load pattern")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.legend(loc="upper right")
    plt.tight_layout()
    out_path = f"{out_dir}/model_accuracy_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")




# ============================================================
# INTERACTIVE MENUS
# ============================================================

def ask(question: str, valid: list, default: str = None) -> str:
    """Prompt until the answer is one of `valid` (case-insensitive)."""
    hint = f" [{default}]" if default else ""
    while True:
        answer = input(f"{question}{hint}: ").strip().lower()
        if not answer and default is not None:
            return default
        if answer in [v.lower() for v in valid]:
            return answer
        print(f"  Please enter one of: {', '.join(valid)}")


def prompt_data_choice() -> list:
    print("\nWhich data do you want to load?")
    print("  1) Attention            (AX-CPT task)")
    print("  2) Cognitive load       (N-back task)")
    print("  3) Both")
    choice = ask("Pick 1, 2 or 3", ["1", "2", "3"], default="3")
    return {"1": ["attention"], "2": ["memory"], "3": ["attention", "memory"]}[choice]


def prompt_ica_choice() -> bool:
    print("\nICA artifact cleaning (removes eye-blink, muscle, heart and channel-noise components):")
    print("  1) Run ICA cleaning   (reuses the cached result from a previous run if one exists)")
    print("  2) Skip ICA           (analyse the uncleaned signal)")
    return ask("Pick 1 or 2", ["1", "2"], default="1") == "1"


def prompt_correlation_choice() -> str:
    print("\nCorrelation analysis (Pearson, each power-band feature vs. hit/miss):")
    print("  1) One experiment you choose  (plus its output / detection plots)")
    print("  2) All experiments pooled together")
    print("  3) Both of the above")
    print("  4) Skip correlation")
    choice = ask("Pick 1-4", ["1", "2", "3", "4"], default="4")
    return {"1": "single", "2": "pooled", "3": "both", "4": "none"}[choice]


def prompt_model_choice() -> list:
    print("\nMachine-learning algorithms (all evaluated with leave-one-subject-out):")
    for i, (_, label) in enumerate(MODEL_MENU, start=1):
        print(f"  {i}) {label}")
    print("  A) Run ALL of the above")
    print("  N) None -- skip machine learning")
    while True:
        answer = input("Pick one or more numbers separated by commas (e.g. 1,3), A or N [A]: ").strip().lower() or "a"
        if answer == "a":
            return list(ALL_MODELS)
        if answer == "n":
            return []
        try:
            picks = [int(p) for p in answer.replace(" ", "").split(",") if p]
            if picks and all(1 <= p <= len(MODEL_MENU) for p in picks):
                return [MODEL_MENU[p - 1][0] for p in dict.fromkeys(picks)]
        except ValueError:
            pass
        print(f"  Enter numbers between 1 and {len(MODEL_MENU)}, or A / N.")


def prompt_phases_choice() -> bool:
    print("\nAlso evaluate the click-locked thinking-window and thinking/decision/motor phase analyses?")
    print("  (needs ICA cleaning; adds several more model runs)")
    return ask("y or n", ["y", "n"], default="y") == "y"


def parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="EEG attention / cognitive-load analysis. Run with no arguments for interactive menus; "
                    "any option given here skips the corresponding menu.")
    p.add_argument("--data", choices=["attention", "cognitive", "both"], help="which data to load")
    p.add_argument("--ica", choices=["run", "skip"], help="ICA artifact cleaning")
    p.add_argument("--correlation", choices=["single", "pooled", "both", "none"], help="correlation analysis")
    p.add_argument("--models", help="comma list of: logistic,logistic_anova,linear,nn,hmm  | all | none")
    p.add_argument("--nn", choices=["1", "2", "3"], help="neural-network architecture")
    p.add_argument("--phases", choices=["yes", "no"], help="also run thinking-window / phase analyses")
    p.add_argument("--experiment", type=int, help="experiment id for the single-experiment correlation plots")
    return p.parse_args()


def collect_options(args) -> dict:
    kinds = {"attention": ["attention"], "cognitive": ["memory"], "both": ["attention", "memory"]}[args.data] \
        if args.data else prompt_data_choice()
    ids_by_kind = discover_local_experiments(kinds)
    for k in kinds:
        if not ids_by_kind[k]:
            raise SystemExit(f"\nNo {k} experiments found in {DATA_ROOT}/ -- see README.md for the data layout.")
    print("\nData loaded.")

    use_ica = (args.ica == "run") if args.ica else prompt_ica_choice()
    correlation = args.correlation if args.correlation else prompt_correlation_choice()

    if args.models:
        models = list(ALL_MODELS) if args.models == "all" else ([] if args.models == "none" else
                                                                  [m.strip() for m in args.models.split(",")])
        bad = [m for m in models if m not in ALL_MODELS]
        if bad:
            raise SystemExit(f"Unknown model(s): {bad}. Choose from {ALL_MODELS}, all, none.")
    else:
        models = prompt_model_choice()

    nn_name, nn_builder = "Neural Network", None
    if "nn" in models:
        if not KERAS_AVAILABLE:
            print("\n[NOTE] tensorflow is not installed -- the neural network will be skipped.")
        else:
            nn_name, nn_builder = (NETWORK_BUILDERS[args.nn] if args.nn else prompt_for_network_choice())

    phases = False
    if models:
        phases = (args.phases == "yes") if args.phases else (prompt_phases_choice() if use_ica else False)

    interactive = not (args.data and args.ica and args.correlation and args.models)
    return {"kinds": kinds, "ids_by_kind": ids_by_kind, "use_ica": use_ica, "correlation": correlation,
            "models": models, "nn_name": nn_name, "nn_builder": nn_builder, "phases": phases,
            "experiment": args.experiment, "interactive": interactive}


def main():
    os.chdir(BASE_DIR)  # results are written next to this script
    args = parse_args()
    opts = collect_options(args)

    print(f"\nRun settings: data={opts['kinds']}, ICA={'run' if opts['use_ica'] else 'skip'}, "
          f"correlation={opts['correlation']}, models={opts['models'] or 'none'}, "
          f"thinking/phase analyses={'yes' if opts['phases'] else 'no'}")

    accuracies = {}
    for kind in opts["kinds"]:
        accuracies[kind] = run_for_kind(opts["ids_by_kind"][kind], kind, opts) or {}

    # Cross-task summary only when both tasks ran with at least one model.
    if len(opts["kinds"]) == 2 and opts["models"] and accuracies["attention"] and accuracies["memory"]:
        summary_table = build_model_summary_table(accuracies["attention"], accuracies["memory"], opts["nn_name"])
        summary_table.to_csv("./model_accuracy_summary.csv", index=False)
        print(f"\nSaved -> ./model_accuracy_summary.csv")
        print(summary_table.to_string(index=False))
        plot_model_summary(summary_table, ".")

    print(f"\n{'#'*70}\nDONE\n{'#'*70}")
    for kind in opts["kinds"]:
        print(f"  {kind} results: {RESULTS_DIR[kind]}/")


if __name__ == "__main__":
    main()
