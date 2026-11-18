"""Track-level Librosa feature extraction and local cache helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def extract_track_features(
    audio_path: str | Path, *, sample_rate: int = 22050, n_mfcc: int = 20
) -> dict[str, float]:
    """Extract frame descriptors and summarize each across one complete recording."""
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - depends on optional runtime install
        raise ImportError("Install librosa to extract audio features.") from exc

    y, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    if y.size == 0:
        raise ValueError(f"Audio file is empty or unreadable: {audio_path}")

    descriptors: dict[str, np.ndarray] = {
        "mfcc": librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc),
        "chroma": librosa.feature.chroma_stft(y=y, sr=sr),
        "spectral_centroid": librosa.feature.spectral_centroid(y=y, sr=sr),
        "spectral_bandwidth": librosa.feature.spectral_bandwidth(y=y, sr=sr),
        "spectral_rolloff": librosa.feature.spectral_rolloff(y=y, sr=sr),
        "zero_crossing_rate": librosa.feature.zero_crossing_rate(y),
        "rms": librosa.feature.rms(y=y),
    }
    features: dict[str, float] = {}
    for name, values in descriptors.items():
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 1:
            array = array[np.newaxis, :]
        for index, row in enumerate(array):
            prefix = f"{name}_{index}" if array.shape[0] > 1 else name
            features[f"{prefix}_mean"] = float(np.mean(row))
            features[f"{prefix}_std"] = float(np.std(row))
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features["tempo"] = float(np.asarray(tempo).reshape(-1)[0])
    if not all(np.isfinite(value) for value in features.values()):
        raise ValueError(f"Non-finite feature value extracted from {audio_path}")
    return features


def build_feature_table(
    tracks: pd.DataFrame, *, sample_rate: int = 22050, n_mfcc: int = 20
) -> pd.DataFrame:
    """Extract one row per recording; failed files are surfaced, never silently skipped."""
    rows = []
    for record in tracks.itertuples(index=False):
        values = extract_track_features(record.path, sample_rate=sample_rate, n_mfcc=n_mfcc)
        rows.append({"track_id": record.track_id, "genre": record.genre, **values})
    return pd.DataFrame(rows)


def save_feature_table(features: pd.DataFrame, path: str | Path) -> None:
    """Persist a local feature cache (the cache directory is git-ignored)."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(destination, index=False)
