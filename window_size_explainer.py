"""
window_size_explainer.py

Standalone explainer for the FFT WINDOW_SIZE trade-off discussed for the
thinking/decision/motor phase split (phase_explainer.py / Step 6c of
run_analysis.py): a shorter window gives better TIME resolution
(each power-band row reflects a narrower slice of the trial) but worse
FREQUENCY resolution (freq_resolution_hz = SFREQ / WINDOW_SIZE), and a
window that's too short can make a band's FFT-bin count hit zero --
silently turning every feature in that band into 0 rather than a real
power estimate.

Note: WINDOW_SIZE is a single constant in run_analysis.py (256
samples / 2000 ms) used identically for BOTH attention and cognitive
load / memory -- it does not differ by task type. (BLOCK_SIZE does differ
-- 32 vs. 64 -- but that's the trial-averaging chunk count, unrelated to
the FFT window used here.)

Produces, for whichever experiment you pick:
  1. A theory table: frequency resolution and FFT-bin count per band,
     for a range of candidate window sizes -- independent of any one
     trial.
  2. A bar plot of that table.
  3. A real, empirical comparison: the SAME example trial's power-band
     trajectory (first principal component), recomputed from the actual
     cleaned EEG at each candidate window size, side by side.

Requires run_analysis.py to have been run at least once first (for
the cached label JSON + cleaned EEG signal this script reads). Needs
only numpy, pandas, matplotlib, scikit-learn.

Run with:  python3 window_size_explainer.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ============================================================
# CONFIG -- must match run_analysis.py, except WINDOW_SIZE_MS_OPTIONS
# which is specific to this explainer (the pipeline itself always uses
# the first entry, 2000 ms / 256 samples).
# ============================================================
SFREQ = 128.0
FINE_STRIDE = 4  # ~31.25 ms step -- held fixed across all window sizes compared here

WINDOW_SIZE_MS_OPTIONS = [2000, 1000, 500, 250, 125]  # 2000 ms = run_analysis.py's current WINDOW_SIZE

EMOTIV_CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2",
                    "P8", "T8", "FC6", "F4", "F8", "AF4"]
BANDS = {"delta": (0.5, 4), "theta": (4, 7), "alpha": (7, 14),
          "beta": (14, 30), "gamma": (30, 45)}

def _find_data_root() -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(base, "data"), base):
        if os.path.isdir(os.path.join(cand, "pb_attention")):
            return cand
    return os.path.join(base, "data")


DATA_ROOT = _find_data_root()
PB_DIR = {"attention": f"{DATA_ROOT}/pb_attention", "memory": f"{DATA_ROOT}/pb_cognitive_load"}
CLEAN_EEG_PB_DIR = "./Clean_Eeg_using_ICA"


# ============================================================
# Core logic
# ============================================================

def load_labels(path: str) -> dict:
    data = json.load(open(path))
    return dict(zip(data[0], data[1]))


def rescale_task_times_to_eeg(match_times, latency_times, false_positives,
                                 detect_success, eeg_duration_ms) -> dict:
    all_raw_times = list(match_times) + list(false_positives)
    hit_click_times_raw, hit_counter = [], 0
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
        "scale_factor": scale_factor,
    }


def reconstruct_all_trials_from_rescaled(rescaled, match_times, detect_success) -> list:
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


def sliding_window_power_bands_at(data: np.ndarray, window_size: int, stride: int) -> pd.DataFrame:
    n_channels, n_total = data.shape
    n_windows = (n_total - window_size) // stride + 1
    freqs = np.fft.rfftfreq(window_size, d=1.0 / SFREQ)
    hann = np.hanning(window_size)
    band_idx = {b: np.logical_and(freqs >= lo, freqs < hi) for b, (lo, hi) in BANDS.items()}
    rows, centers = [], []
    for w in range(n_windows):
        start = w * stride
        window = data[:, start:start + window_size]
        row = {}
        for ci, ch in enumerate(EMOTIV_CHANNELS):
            x = window[ci]
            detrended = x - np.polyval(np.polyfit(np.arange(len(x)), x, 1), np.arange(len(x)))
            power = np.abs(np.fft.rfft(detrended * hann)) ** 2
            for band in BANDS:
                row[f"{band}_{ch.lower()}"] = power[band_idx[band]].sum()
        rows.append(row)
        centers.append((start + window_size / 2) / SFREQ * 1000)
    df = pd.DataFrame(rows)
    df["center_ms"] = centers
    return df


def bin_count_table(window_sizes_ms: list) -> pd.DataFrame:
    """Theory table -- doesn't need any trial data, just the math:
    freq_resolution_hz = SFREQ / window_size, and how many FFT bins
    that puts inside each band."""
    rows = []
    for wms in window_sizes_ms:
        ws = int(round(wms / 1000 * SFREQ))
        freq_res = SFREQ / ws
        freqs = np.fft.rfftfreq(ws, d=1.0 / SFREQ)
        row = {"window_size_ms": wms, "window_size_samples": ws, "freq_resolution_hz": freq_res}
        for band, (lo, hi) in BANDS.items():
            row[f"{band}_bins"] = int(np.sum((freqs >= lo) & (freqs < hi)))
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# Interactive prompts (same as phase_explainer.py)
# ============================================================

def prompt_for_kind() -> str:
    print("\nWhich task do you want to explain?")
    print("  1) Attention")
    print("  2) Cognitive load / memory")
    print("  (note: WINDOW_SIZE is the SAME constant -- 256 samples / 2000 ms -- for both "
          "tasks in run_analysis.py; only BLOCK_SIZE differs, 32 vs. 64, and that's "
          "an unrelated trial-averaging chunk count, not the FFT window)")
    while True:
        choice = input("Pick 1 or 2: ").strip()
        if choice == "1":
            return "attention"
        if choice == "2":
            return "memory"
        print("  Please enter 1 or 2.")


def available_experiments(kind: str) -> list:
    pb_dir = PB_DIR[kind]
    ids = []
    if not os.path.isdir(pb_dir):
        return ids
    for fname in sorted(os.listdir(pb_dir)):
        if not fname.endswith(f"-{kind}.json"):
            continue
        exp_id = fname.split(f"-{kind}.json")[0]
        eeg_path = f"{CLEAN_EEG_PB_DIR}/{exp_id}_{kind}_cleaned_eeg.csv"
        if os.path.exists(eeg_path):
            ids.append(int(exp_id))
    return sorted(ids)


def prompt_for_experiment(kind: str) -> int:
    ids = available_experiments(kind)
    if not ids:
        raise SystemExit(f"No usable experiments found for '{kind}'. Run run_analysis.py "
                           f"first so {PB_DIR[kind]}/ and {CLEAN_EEG_PB_DIR}/ are populated.")
    print(f"\nAvailable {kind} experiments (label data + cached cleaned EEG both present):")
    print(", ".join(str(i) for i in ids))
    while True:
        raw = input("Pick an experiment ID: ").strip()
        if raw.isdigit() and int(raw) in ids:
            return int(raw)
        print("  Please enter one of the listed experiment IDs.")


# ============================================================
# Plots
# ============================================================

def plot_bin_counts(table: pd.DataFrame, exp_id, kind, out_dir):
    """No figure title -- it's in the filename. Bar labels are kept
    (they're short axis-level labels, not a title)."""
    band_cols = list(BANDS.keys())
    x = np.arange(len(table))
    width = 0.15
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, band in enumerate(band_cols):
        ax.bar(x + i * width, table[f"{band}_bins"], width, label=band)
    ax.set_xticks(x + width * (len(band_cols) - 1) / 2)
    ax.set_xticklabels([f"{w}ms\n({r:.2f}Hz)" for w, r in
                          zip(table["window_size_ms"], table["freq_resolution_hz"])])
    ax.set_xlabel("window size (freq. resolution)")
    ax.set_ylabel("FFT bins inside the band")
    ax.axhline(0, color="gray", lw=0.8)
    ax.legend()
    fig.tight_layout()
    out_path = f"{out_dir}/fft_bins_per_band_vs_window_size_{exp_id}_{kind}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def plot_trajectory_comparison(trial_idx, stim, click, window_sizes_ms, trajectories, exp_id, kind, out_dir):
    """No global figure title -- it's in the filename. Each panel keeps
    a short caption (window size + freq resolution + row count) since
    that's needed to tell the panels apart within one image."""
    fig, axes = plt.subplots(1, len(window_sizes_ms), figsize=(5 * len(window_sizes_ms), 4.5), sharey=False)
    if len(window_sizes_ms) == 1:
        axes = [axes]
    for ax, wms in zip(axes, window_sizes_ms):
        rel_ms, pc1, freq_res, n = trajectories[wms]
        if pc1 is None:
            ax.text(0.5, 0.5, f"window={wms}ms: only {n} row(s), too few", ha="center", va="center",
                     transform=ax.transAxes, fontsize=9)
            continue
        ax.plot(rel_ms, pc1, "o-", color="#55A868")
        ax.set_title(f"window={wms}ms (freq res={freq_res:.2f}Hz), n={n} rows", fontsize=10)
        ax.set_xlabel("ms since stimulus onset")
    axes[0].set_ylabel("PC1 score")
    fig.tight_layout()
    out_path = (f"{out_dir}/window_size_trajectory_comparison_trial{trial_idx}"
                 f"_experiment_{exp_id}_{kind}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # data lives next to this script
    kind = prompt_for_kind()
    exp_id = prompt_for_experiment(kind)

    out_dir = f"./window_size_explainer_output/{kind}_experiment_{exp_id}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nSaving everything to: {out_dir}/")

    # --- 1. theory table (no trial data needed) ---
    table = bin_count_table(WINDOW_SIZE_MS_OPTIONS)
    table.to_csv(f"{out_dir}/frequency_resolution_bin_counts_{exp_id}_{kind}.csv", index=False)
    print(f"\n  Saved -> {out_dir}/frequency_resolution_bin_counts_{exp_id}_{kind}.csv")
    print(table.to_string(index=False))
    plot_bin_counts(table, exp_id, kind, out_dir)

    # --- load label data + cleaned EEG, rescale, reconstruct trials ---
    label_path = f"{PB_DIR[kind]}/{exp_id}-{kind}.json"
    eeg_path = f"{CLEAN_EEG_PB_DIR}/{exp_id}_{kind}_cleaned_eeg.csv"
    rec = load_labels(label_path)
    match_times = rec["match_times"]
    latency_times = rec.get("latency_times") or []
    false_positives = rec.get("false_positives") or []
    detect_success = rec["detectSuccess"]

    eeg_df = pd.read_csv(eeg_path)
    data = eeg_df[EMOTIV_CHANNELS].values.T
    eeg_duration_ms = data.shape[1] / SFREQ * 1000

    rescaled = rescale_task_times_to_eeg(match_times, latency_times, false_positives,
                                            detect_success, eeg_duration_ms)
    trials = reconstruct_all_trials_from_rescaled(rescaled, match_times, detect_success)

    # --- pick the same kind of example trial as phase_explainer.py: the
    # hit closest to the median hit reaction time ---
    hit_trials = [t for t in trials if t["outcome"] == 1]
    rts = [t["click_time_ms"] - t["stimulus_onset_ms"] for t in hit_trials]
    median_rt = float(np.median(rts))
    example = min(hit_trials, key=lambda t: abs((t["click_time_ms"] - t["stimulus_onset_ms"]) - median_rt))
    trial_idx = example["trial_idx"]
    stim, click = example["stimulus_onset_ms"], example["click_time_ms"]
    print(f"\nExample trial: trial_idx={trial_idx} (RT={click - stim:.1f} ms, "
          f"closest to the median hit RT of {median_rt:.1f} ms)")

    # --- 2. empirical comparison: recompute this one trial's power bands
    # at every candidate window size, directly from the cleaned EEG ---
    trajectories = {}
    all_rows = []
    for wms in WINDOW_SIZE_MS_OPTIONS:
        ws = int(round(wms / 1000 * SFREQ))
        pb = sliding_window_power_bands_at(data, ws, FINE_STRIDE)
        feature_cols = [c for c in pb.columns if c != "center_ms"]
        mask = (pb["center_ms"] >= stim) & (pb["center_ms"] < click)
        sub = pb.loc[mask].reset_index(drop=True)
        n = len(sub)
        freq_res = SFREQ / ws
        if n < 4:
            trajectories[wms] = (None, None, freq_res, n)
            print(f"  window={wms}ms: only {n} row(s) in this trial, too few to show a trajectory")
            continue
        Xz = StandardScaler().fit_transform(sub[feature_cols].values)
        n_comp = min(8, Xz.shape[0] - 1, Xz.shape[1])
        Xp = PCA(n_components=n_comp, random_state=42).fit_transform(Xz)
        rel_ms = sub["center_ms"].values - stim
        trajectories[wms] = (rel_ms, Xp[:, 0], freq_res, n)

        sub_out = sub.copy()
        sub_out.insert(0, "window_size_ms", wms)
        all_rows.append(sub_out)
        print(f"  window={wms}ms: freq res={freq_res:.2f}Hz, {n} rows in this trial")

    if all_rows:
        pd.concat(all_rows, ignore_index=True).to_csv(
            f"{out_dir}/example_trial{trial_idx}_power_bands_by_window_size_{exp_id}_{kind}.csv", index=False)
        print(f"\n  Saved -> {out_dir}/example_trial{trial_idx}_power_bands_by_window_size_{exp_id}_{kind}.csv")

    plot_trajectory_comparison(trial_idx, stim, click, WINDOW_SIZE_MS_OPTIONS, trajectories, exp_id, kind, out_dir)

    print(f"\nDone. Everything is in: {out_dir}/")


if __name__ == "__main__":
    main()
