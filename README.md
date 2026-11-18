# Music Genre Classification

A reproducible audio-classification project comparing **track-level tabular audio features** with a **log-Mel spectrogram CNN** on the GTZAN music-genre dataset.

> Status: project scaffold and experiment design. Model results will be added only after the notebook has been run and the saved outputs have been inspected.

## Objective

Classify a complete audio recording into one of ten music genres. The primary comparison will use the same recording-level train/validation/test split for both approaches:

1. **Tabular baseline:** extract Librosa descriptors (MFCCs, chroma, spectral, energy and rhythm features), aggregate each descriptor across time, then train Scikit-learn classifiers.
2. **Spectrogram model:** convert each complete track into a fixed-size log-Mel representation and train a compact CNN.

The optional bonus experiment (transfer learning) will be considered only after the two baselines work reliably and within the available free Colab resources.

## Dataset

Recommended source: [GTZAN Genre Collection on Kaggle](https://www.kaggle.com/datasets/carlthome/gtzan-genre-collection).

Download the dataset through Kaggle/Google Colab and place/extract it under `data/`. The loader supports the common directory layout `data/genres_original/<genre>/*.wav` and can be pointed at another extracted root with `MUSIC_DATA_DIR`.

**Do not commit audio files or derived datasets.** Dataset versions can differ; the notebook will audit the actual files, labels, durations, sample rates, unreadable files and duplicate candidates before training. GTZAN is a small, historically used benchmark with known dataset-quality limitations, so results will be interpreted cautiously rather than treated as definitive real-world performance.

## Reproducibility and evaluation rules

- Split by **original recording**, never by spectrogram frame or audio segment. All representations from one recording must remain in one split.
- Use one fixed, stratified split for both approaches (seed configured in `configs/config.yaml`).
- Fit learned preprocessing only on training data. Do not use the test set for model selection.
- Report accuracy and macro-F1 as primary metrics, plus per-class precision/recall/F1, confusion matrices and training configuration.
- Compare models on the exact same held-out test recordings.
- Cache extracted features/representations locally to avoid repeating expensive audio decoding. Caches and generated model files are ignored by Git.
- Record library versions, random seeds, split counts and dataset audit findings in the notebook/report.

## Repository layout

```
configs/config.yaml                         experiment settings
notebooks/task_06_music_genre_classification.ipynb
src/music_genre_classification/             reusable project code
tests/                                      lightweight unit tests
reports/                                    generated summaries/figures (no raw data)
```

## Setup (Google Colab or local Python)

Python 3.10–3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
pip install -e .
pytest
```

In Colab, install the requirements in a notebook cell and enable a GPU runtime for the CNN experiment if available. A GPU is helpful but not required for the tabular baseline.

## Dataset location

Set `MUSIC_DATA_DIR` to the directory containing `genres_original`, or update `dataset_root` in the config. Example:

```bash
export MUSIC_DATA_DIR=/path/to/gtzan
```

## Planned notebook workflow

1. Environment, hardware and reproducibility checks.
2. Dataset discovery and quality audit.
3. Recording-level stratified split and split integrity checks.
4. Librosa feature extraction and cache creation.
5. Scikit-learn baseline training and validation-based selection.
6. Log-Mel spectrogram creation and track-level CNN training.
7. Same-test-set evaluation, class-wise diagnostics and comparison.
8. Error analysis, limitations, reproducibility summary and conclusions.

## Current implementation status

- [x] Initial repository structure and experiment protocol
- [x] Dataset discovery and feature-extraction utilities
- [x] Track-level split and evaluation helpers
- [ ] Run dataset audit in Colab
- [ ] Train and evaluate tabular models
- [ ] Implement/train spectrogram CNN
- [ ] Compare models and complete results/report
- [ ] Final notebook, tests and documentation audit

No performance numbers are claimed until the experiments have actually run.
