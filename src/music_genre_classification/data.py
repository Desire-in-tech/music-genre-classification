"""Dataset discovery and recording-level split utilities."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".au"}


def discover_tracks(dataset_root: str | Path | None = None) -> pd.DataFrame:
    """Discover audio files under genres_original/<genre>/ and infer genre labels."""
    root = Path(dataset_root or os.getenv("MUSIC_DATA_DIR", "data")).expanduser()
    audio_root = root / "genres_original"
    if not audio_root.is_dir():
        # Also accept a path that directly points at genres_original.
        if root.name == "genres_original" and root.is_dir():
            audio_root = root
        else:
            raise FileNotFoundError(
                f"Could not find 'genres_original' under {root}. "
                "Set MUSIC_DATA_DIR or dataset.dataset_root to the extracted GTZAN root."
            )

    rows = [
        {"track_id": f"{genre_dir.name}/{path.name}", "path": str(path), "genre": genre_dir.name}
        for genre_dir in sorted(audio_root.iterdir()) if genre_dir.is_dir()
        for path in sorted(genre_dir.iterdir())
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]
    if not rows:
        raise ValueError(f"No supported audio files found in {audio_root}")
    frame = pd.DataFrame(rows).sort_values("track_id").reset_index(drop=True)
    if frame["track_id"].duplicated().any():
        raise ValueError("Duplicate track identifiers detected; ensure genre folders are unique.")
    return frame


def split_tracks(
    tracks: pd.DataFrame,
    *,
    test_size: float = 0.20,
    validation_size: float = 0.20,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train/validation/test recording splits with stratified genre labels.

    validation_size is the fraction of the remaining development set, not of all data.
    All derived audio representations must inherit these track IDs to prevent leakage.
    """
    required = {"track_id", "genre"}
    missing = required.difference(tracks.columns)
    if missing:
        raise ValueError(f"tracks is missing required columns: {sorted(missing)}")
    if not 0 < test_size < 1 or not 0 < validation_size < 1:
        raise ValueError("test_size and validation_size must both be between 0 and 1.")
    if tracks["track_id"].duplicated().any():
        raise ValueError("Each recording must have a unique track_id before splitting.")

    development, test = train_test_split(
        tracks, test_size=test_size, random_state=random_seed, stratify=tracks["genre"]
    )
    train, validation = train_test_split(
        development, test_size=validation_size, random_state=random_seed,
        stratify=development["genre"]
    )
    return tuple(part.sort_values("track_id").reset_index(drop=True)
                 for part in (train, validation, test))
