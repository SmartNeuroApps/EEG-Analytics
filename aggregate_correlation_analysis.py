"""
aggregate_correlation_analysis.py

Pearson correlation between each power-band feature and hit/miss outcome,
computed on the AVERAGED dataset ACROSS ALL EXPERIMENTS of a task type --
not the single-experiment correlation already produced by Step 4 of
run_analysis.py. Runs automatically for BOTH attention and cognitive
load / memory, no prompts needed, since the point is to look at everything
pooled together.

For each task type, produces:
  - a CSV table: sensor, correlation, p_value (one row per power-band
    feature, i.e. {band}_{electrode} -- 70 rows: 5 bands x 14 electrodes)
  - a correlation-VALUE bar plot (one bar per feature, grouped/colored by
    band)
  - a SEPARATE p-value bar plot (never combined with the correlation-value
    plot, matching the rest of this project's convention -- a 0.05
    reference line is drawn, but note 70 features are tested here, so a
    naive p<0.05 cutoff will produce some false positives by chance;
    this is not a multiple-comparisons correction)
  - a heatmap of the FULL feature-by-feature correlation matrix (every
    power-band feature against every other, not just against outcome) --
    this is the "heatmap" companion to the two bar plots above

Reads the cleaned, trial-averaged dataset that run_analysis.py's
Step 2/Step 7 already caches (clean_data_attention/cleaned_power_bands.csv,
clean_data_memory/cleaned_power_bands.csv) -- run run_analysis.py at
least once first so these exist. Needs numpy, pandas, matplotlib, seaborn,
scipy.

Plots have no in-figure title; what each shows is in its filename.

Run with:  python3 aggregate_correlation_analysis.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

CLEAN_DATA_DIR = {"attention": "./clean_data_attention", "memory": "./clean_data_memory"}
OUT_DIR_ROOT = "./aggregate_correlation_output"

BCI_ELECTRODES = ["af3", "af4", "f3", "f4", "f7", "f8", "fc5", "fc6",
                    "o1", "o2", "p7", "p8", "t7", "t8"]

# Band-major sensor ordering/coloring, matching the layout used by the
# original CorrelationAnalysis.py / correlation_after_cleaning.py aggregate
# plots and the single-experiment Pearson plots in run_analysis.py.
BAND_PLOT_ORDER = ["alpha", "beta", "delta", "gamma", "theta"]


def canonical_sensor_order(columns) -> list:
    columns = list(columns)
    ordered = [f"{band}_{e}" for band in BAND_PLOT_ORDER for e in BCI_ELECTRODES if f"{band}_{e}" in columns]
    leftover = [c for c in columns if c not in ordered]
    return ordered + leftover


def sensor_band_colors(sensor_names) -> list:
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    band_color = {band: cycle[i % len(cycle)] for i, band in enumerate(BAND_PLOT_ORDER)}
    return [band_color.get(str(s).split("_")[0], "#888888") for s in sensor_names]


def load_cleaned_data(kind: str):
    """Reads ONLY the post-ICA cleaned dataset -- never the uncleaned
    power bands. This file is written by run_analysis.py's Step 7,
    right after Step 2 runs ICA artifact rejection on every experiment
    (or loads an already-ICA-cleaned cache), so its existence IS the
    guarantee that ICA has run for this task type. There's no fallback
    to uncleaned data here on purpose: this script is read-only on
    run_analysis.py's output, not an alternate way to get
    correlations without cleaning the data first."""
    path = f"{CLEAN_DATA_DIR[kind]}/cleaned_power_bands.csv"
    if not os.path.exists(path):
        raise SystemExit(
            f"\n{path} not found.\n"
            f"This script only runs on the ICA-CLEANED dataset, which run_analysis.py writes "
            f"after running ICA artifact rejection (Step 2) and saving the result (Step 7).\n"
            f"Run `python run_analysis.py` first (through at least one full pass for "
            f"'{kind}') so {path} exists, then run this script again."
        )
    df = pd.read_csv(path)
    y = df["Sucess"].to_numpy()
    X = df.drop(columns=["Sucess", "subject", "exp_id"])
    return X, y


def compute_pearson_table(X: pd.DataFrame, y: np.ndarray, kind: str, out_dir: str) -> pd.DataFrame:
    """Same computation as run_analysis.py's
    compute_pearson_table_single_experiment(), but X/y here are pooled
    across EVERY experiment of this kind, not just one."""
    ordered_cols = canonical_sensor_order(X.columns)
    X = X[ordered_cols]

    rows = []
    for col in X.columns:
        try:
            corr, p_value = stats.pearsonr(X[col], y)
        except Exception:
            corr, p_value = np.nan, np.nan
        rows.append({"sensor": col, "correlation": corr, "p_value": p_value})

    table = pd.DataFrame(rows)
    out_path = f"{out_dir}/pearson_correlation_table_all_experiments_{kind}.csv"
    table.to_csv(out_path, index=False)
    print(f"  Saved -> {out_path}")
    return table


def plot_pearson_correlation(table: pd.DataFrame, kind: str, n_experiments: int, n_trials: int, out_dir: str):
    colors = sensor_band_colors(table["sensor"])
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(range(len(table)), table["correlation"], color=colors)
    ax.set_xticks(range(len(table)))
    ax.set_xticklabels(table["sensor"], rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Sensors")
    ax.set_ylabel("Correlation (Pearson)")
    plt.tight_layout()
    out_path = f"{out_dir}/pearson_correlation_all_experiments_{kind}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def plot_pearson_pvalues(table: pd.DataFrame, kind: str, n_experiments: int, n_trials: int, out_dir: str):
    colors = sensor_band_colors(table["sensor"])
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(range(len(table)), table["p_value"], color=colors)
    ax.axhline(0.05, color="red", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(table)))
    ax.set_xticklabels(table["sensor"], rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Sensors")
    ax.set_ylabel("p-value (Pearson)")
    plt.tight_layout()
    out_path = f"{out_dir}/pearson_pvalues_all_experiments_{kind}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def plot_correlation_heatmap(X: pd.DataFrame, kind: str, out_dir: str):
    """Full feature-by-feature correlation matrix -- every power-band
    feature against every other, pooled across every experiment of this
    kind. No title, matching the single-experiment heatmap's convention."""
    ordered_cols = canonical_sensor_order(X.columns)
    correlation_matrix = X[ordered_cols].corr()
    fig = plt.figure(figsize=(20, 16))
    sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", fmt=".2f")
    out_path = f"{out_dir}/sensor_correlation_heatmap_all_experiments_{kind}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def run_for_kind(kind: str):
    task_label = "attention" if kind == "attention" else "cognitive load / memory"
    print(f"\n{'#'*70}\n{task_label.upper()}: aggregate Pearson correlation, all experiments combined\n{'#'*70}")

    X, y = load_cleaned_data(kind)
    if X is None:
        return

    out_dir = f"{OUT_DIR_ROOT}/{kind}"
    os.makedirs(out_dir, exist_ok=True)

    cleaned_df = pd.read_csv(f"{CLEAN_DATA_DIR[kind]}/cleaned_power_bands.csv")
    n_experiments = cleaned_df["exp_id"].nunique()
    n_trials = len(X)
    print(f"  {n_trials} trials pooled across {n_experiments} experiments")

    table = compute_pearson_table(X, y, kind, out_dir)
    plot_pearson_correlation(table, kind, n_experiments, n_trials, out_dir)
    plot_pearson_pvalues(table, kind, n_experiments, n_trials, out_dir)
    plot_correlation_heatmap(X, kind, out_dir)


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # data lives next to this script
    for kind in ("attention", "memory"):
        run_for_kind(kind)
    print(f"\nDone. Everything is in: {OUT_DIR_ROOT}/")


if __name__ == "__main__":
    main()
