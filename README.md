# EEG Attention & Cognitive-Load Analysis

Code and anonymized data accompanying **"A Software Architecture for EEG-Enabled Business Applications"** (EMOTIV EPOC, 14 channels, 128 Hz).

The pipeline cleans the EEG with ICA, computes power-band features (delta, theta, alpha, beta, gamma on 14 channels = 70 features), analyses their correlation with task performance, and predicts hit/miss with five models under leave-one-out evaluation.

Everything runs locally. Nothing is downloaded and no server is contacted.

## Quick start

```bash
git clone <this-repo>
cd eeg-attention-cognitive-load
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_analysis.py
```

The script reads the data in `./data` (next to the script) and asks you:

1. **Which data to load**: attention, cognitive load, or both.
2. **ICA cleaning**: run it (infomax + ICLabel; the cleaned data is cached in `clean_data_*/` and reused on later runs) or skip it.
3. **Correlation analysis**: one experiment, all experiments pooled, both, or none.
4. **Machine-learning models**: all at once, or any subset of Logistic Regression, Logistic + ANOVA, Linear Regression, Neural Network (3 architectures), HMM.
5. **Thinking-window and phase analyses** (only offered when ICA is run).

### Non-interactive use

```bash
python run_analysis.py --data both --ica run --correlation both --models all --nn 2 --phases no
python run_analysis.py --data attention --ica skip --correlation pooled --models logistic,hmm
python run_analysis.py --help
```

| Flag | Values |
|---|---|
| `--data` | `attention`, `cognitive`, `both` |
| `--ica` | `run`, `skip` |
| `--correlation` | `single`, `pooled`, `both`, `none` |
| `--models` | `all`, `none`, or a comma list of `logistic,logistic_anova,linear,nn,hmm` |
| `--nn` | `1`, `2`, `3` (network architecture) |
| `--phases` | `yes`, `no` |
| `--experiment` | experiment id for single-experiment plots |

ICA and the neural network are the slow parts. A first run with ICA on all 27 + 27 recordings takes a while; later runs reuse the cache.

## Data layout

```
data/
  participants.csv                 participant, task, experiment_id
  pb_attention/        {id}-attention.json   {id}-pb.json
  eeg_attention/       {id}-attention-eeg.json
  pb_cognitive_load/   {id}-memory.json      {id}-pb.json
  eeg_cognitive_load/  {id}-memory-eeg.json
```

- 27 attention recordings and 27 cognitive-load recordings.
- Participants are labelled `BrainSoft Participant 1` to `27` per task, numbered by experiment id. All names were removed.
- Attention experiment 178 and cognitive-load experiment 181 are excluded so both tasks have 27 recordings.

## Outputs

Plots and CSVs are written next to the script:

- `attention_results/` and `cognitive_load_results/`: per-experiment plots (output, match-time vs. attention, Pearson correlation, p-values, sensor heatmap), ICA component-exclusion and channel-effect plots, clean-vs-uncleaned model comparison, accuracy per experiment, and the thinking-window and phase CSVs.
- `model_accuracy_summary.csv/.png`: cross-task summary, written when both tasks were run with models.

Figures carry no in-figure titles; the title is in the file name.

`example_results/` holds sample output for one experiment per task (attention 190, cognitive load 200). These figures come from an earlier run that grouped participants by name, so they are illustrative. Numbers from this release can differ.

## Extra scripts

Both run after `run_analysis.py` has been run with ICA at least once (they read its caches) and need no prompts beyond the choices shown.

- `aggregate_correlation_analysis.py`: Pearson correlation of each power-band feature with hit/miss, pooled across all experiments of a task, for both tasks automatically. Writes a table, a correlation plot, a separate p-value plot and a feature-by-feature heatmap to `aggregate_correlation_output/`. Needs `clean_data_*/cleaned_power_bands.csv`.
- `window_size_explainer.py`: shows how FFT window length (2000, 1000, 500, 250, 125 ms) trades frequency resolution against time resolution on one example trial. Asks for the task and an experiment id. Needs `Clean_Eeg_using_ICA/`. Writes to `window_size_explainer_output/`.

## Evaluation: please read

Every model is evaluated leave-one-group-out. Each group is held out in turn; the scaler, ANOVA selector, linear threshold (Youden's J), HMM state mapping and neural network are fit on the remaining groups only. Reported accuracy and balanced accuracy pool the held-out predictions.

In this release the group is the **recording** (`BrainSoft Participant N`). Several recordings come from the same person, so this is **leave-one-recording-out**: the same person's other recordings stay in the training set. The anonymization removed the identity link, so a strict leave-one-person-out test cannot be reproduced from the released data. Expect results to differ from the paper's figures, and treat this as a more optimistic estimate than person-independent evaluation.

## Models

| Key | Model |
|---|---|
| `logistic` | Logistic Regression |
| `logistic_anova` | Logistic Regression with ANOVA feature selection (k = 14) |
| `linear` | Linear Regression, sigmoid output, Youden's J threshold |
| `nn` | Neural Network (3 architectures) |
| `hmm` | 2-state Gaussian HMM with majority-vote state-to-label mapping |

## Preprocessing

- Band-pass 1-45 Hz FIR and average reference.
- ICA: infomax, 10 components, ICLabel. Components not labelled "brain" or "other" are removed.
- Power bands from a 256-sample FFT window (stride 16), averaged per trial.
- Task times are rescaled to EEG time. The thinking window is [stimulus, click - 200 ms); the motor window is click +/- 200 ms.

## Requirements

Python 3.10 or newer. See `requirements.txt` (numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, mne, mne-icalabel, onnxruntime, hmmlearn, tensorflow). TensorFlow is optional; without it the Neural Network option is hidden.

## Citation

If you use this code or data, please cite the paper above.
