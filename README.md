# Music Genre Classification

A reproducible audio-classification project comparing **track-level tabular audio features** with a **log-Mel spectrogram CNN** on the GTZAN music-genre dataset.

## Open the notebook in Google Colab

[Open Task 6 notebook in Google Colab](https://colab.research.google.com/github/Desire-in-tech/music-genre-classification/blob/task-06/music-genre-classification/notebooks/task_06_music_genre_classification.ipynb)

The notebook is designed to run from top to bottom in Colab. Select a GPU runtime for the CNN if one is available. It clones this working branch, installs dependencies, downloads GTZAN through KaggleHub, audits the audio, extracts/caches features, trains both model families, evaluates them on the same held-out recordings, and saves reports/models in the Colab runtime.

If KaggleHub requests authentication, create a Kaggle API token, upload `kaggle.json` when prompted, and do not commit that token. Dataset files, caches, and generated models are intentionally excluded from Git.

## Objective and approaches

Classify complete recordings into music genres and compare:

1. **Tabular baseline:** Librosa MFCC, chroma, spectral, energy and rhythm descriptors aggregated across each complete recording; Scikit-learn RBF-SVM, Random Forest and Extra Trees.
2. **Spectrogram CNN:** fixed-size log-Mel spectrogram for each complete recording; compact TensorFlow CNN with validation-based early stopping.

Transfer learning is an optional extension, not a prerequisite. It is intentionally excluded from the core run so the required approaches can be completed and evaluated without adding unnecessary compute or complexity.

## Dataset

Recommended source: [GTZAN Genre Collection on Kaggle](https://www.kaggle.com/datasets/carlthome/gtzan-genre-collection).

The loader searches for the `genres_original/<genre>/*` directory, including common nested extraction layouts. Dataset contents are audited at runtime rather than assuming every mirror has identical counts or files.

GTZAN is a relatively small, historically used benchmark with known data-quality concerns. Exact SHA-256 duplicates are audited, but this does not identify near-duplicates, alternate encodings, artist overlap or every possible label issue. Interpret benchmark scores cautiously.

## Experimental protocol

- Audit exact SHA-256 duplicates before splitting. Keep one representative of same-label byte-identical recordings and quarantine every conflicting-label duplicate group; then split by unique original recording. This prevents exact-content leakage, though near-duplicates and artist overlap remain possible.
- Use one fixed stratified train/validation/test manifest for both model families (seed 42).
- Fit imputation/scaling only on training data; select models using validation macro-F1.
- Refit the selected tabular estimator on train + validation, then evaluate on the held-out test set. CNN early stopping uses validation only.
- Compare both approaches on exactly the same test recording IDs.
- Report accuracy, macro precision/recall/F1, per-class metrics and confusion matrices.
- Cache derived features locally to avoid unnecessary re-extraction. Generated data, caches and model files are not committed.

## Repository layout

```
configs/config.yaml
notebooks/task_06_music_genre_classification.ipynb
src/music_genre_classification/
tests/
reports/
```

## Local setup

Python 3.10–3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
pip install -e .
pytest -q
```

## Implementation

- [x] Dataset discovery, audio metadata audit and exact-file duplicate check
- [x] Recording-level stratified split utilities
- [x] Librosa tabular feature extraction and local cache
- [x] Scikit-learn baseline model pipelines
- [x] Fixed-size log-Mel representation and CNN builder
- [x] Multiclass evaluation helpers
- [x] End-to-end Colab notebook for both approaches
- [x] Unit tests and GitHub Actions test workflow
- [x] Execute notebook on the actual Kaggle dataset and review saved outputs/results

## Latest recorded experiment

The committed Colab notebook was executed end to end on the Kaggle GTZAN collection. All 13 code cells have saved outputs, with no recorded cell errors.

| Experiment | Test accuracy | Test macro-F1 |
|---|---:|---:|
| Extra Trees (selected tabular baseline) | 72.86% | 72.36% |
| Log-Mel spectrogram CNN | 64.82% | 63.62% |

The dataset audit found 1,000 original recordings and no unreadable files. Exact byte-identical duplicate handling retained 992 unique recordings: six same-label duplicate copies were removed and the two recordings in one conflicting-label group were quarantined. The shared stratified split contains 634 training, 159 validation and 199 test recordings.

The CNN's earlier single-class prediction failure was corrected. Its final test predictions cover all ten genres. Extra Trees remains the stronger model on this split; the CNN is retained as the deep-learning comparison rather than presented as the best-performing approach.

These are results from one fixed split and one dataset mirror, not a claim of broad generalization. GTZAN's known limitations remain relevant: the audit detects exact byte-level duplicates, but does not rule out near-duplicates, alternate encodings, artist overlap or other benchmark biases. The notebook saves detailed metrics, predictions, duplicate-resolution and split reports, plus trained model files, in the Colab runtime under `artifacts/`; generated data and model artifacts are intentionally not committed.
