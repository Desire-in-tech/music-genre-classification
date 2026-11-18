"""Dataset discovery, auditing and duplicate-safe recording-level splits."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd
import soundfile as sf
from sklearn.model_selection import train_test_split

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".au"}


def _find_audio_root(root: Path) -> Path:
    if root.name == "genres_original" and root.is_dir():
        return root
    candidates = [root / "genres_original", root / "Data" / "genres_original", root / "GTZAN" / "genres_original"]
    candidates += sorted(root.glob("**/genres_original"))
    for candidate in candidates:
        if candidate.is_dir() and any(path.is_dir() for path in candidate.iterdir()):
            return candidate
    raise FileNotFoundError(
        f"Could not locate a genres_original directory under {root}. "
        "Set MUSIC_DATA_DIR to the extracted GTZAN folder."
    )


def discover_tracks(dataset_root: str | Path | None = None) -> pd.DataFrame:
    root = Path(dataset_root or os.getenv("MUSIC_DATA_DIR", "data")).expanduser()
    audio_root = _find_audio_root(root)
    rows = []
    for genre_dir in sorted(path for path in audio_root.iterdir() if path.is_dir()):
        for path in sorted(
            path for path in genre_dir.iterdir()
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        ):
            rows.append({
                "track_id": f"{genre_dir.name}/{path.name}",
                "path": str(path),
                "genre": genre_dir.name,
            })
    if not rows:
        raise ValueError(f"No supported audio files found in {audio_root}")
    frame = pd.DataFrame(rows).sort_values("track_id").reset_index(drop=True)
    if frame.track_id.duplicated().any():
        raise ValueError("Duplicate track identifiers detected.")
    return frame


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_tracks(tracks: pd.DataFrame, hash_files: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-file metadata and all rows participating in exact-content duplicates."""
    rows = []
    for record in tracks.itertuples(index=False):
        row = {"track_id": record.track_id, "genre": record.genre, "path": record.path}
        try:
            info = sf.info(record.path)
            row.update(
                duration_seconds=float(info.duration),
                sample_rate=int(info.samplerate),
                channels=int(info.channels),
                frames=int(info.frames),
                readable=True,
            )
            if hash_files:
                row["sha256"] = sha256_file(record.path)
        except Exception as exc:
            row.update(readable=False, error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    audit = pd.DataFrame(rows)
    duplicates = (
        audit[audit.sha256.duplicated(keep=False)].sort_values("sha256")
        if hash_files and "sha256" in audit
        else pd.DataFrame()
    )
    return audit, duplicates


def prepare_modeling_tracks(
    tracks: pd.DataFrame, audit: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove exact duplicate copies and quarantine identical audio with conflicting labels.

    For same-label exact duplicates, retain the lexicographically first track ID.
    If identical bytes have multiple genre labels, exclude the entire hash group because
    there is no defensible single supervised target for that audio.
    """
    if not {"track_id", "genre"}.issubset(tracks.columns):
        raise ValueError("tracks must contain track_id and genre.")
    if not {"track_id", "sha256"}.issubset(audit.columns):
        raise ValueError("audit must contain track_id and sha256; run audit_tracks(hash_files=True).")
    if tracks.track_id.duplicated().any() or audit.track_id.duplicated().any():
        raise ValueError("track_id must be unique in tracks and audit.")
    merged = tracks.merge(audit[["track_id", "sha256"]], on="track_id", how="left", validate="one_to_one")
    if merged.sha256.isna().any():
        missing = merged.loc[merged.sha256.isna(), "track_id"].tolist()
        raise ValueError(f"Missing SHA-256 hashes for {len(missing)} recordings: {missing[:5]}")

    keep_ids = set()
    report_rows = []
    for digest, group in merged.groupby("sha256", sort=True):
        group = group.sort_values("track_id")
        ids = group.track_id.tolist()
        genres = sorted(group.genre.astype(str).unique().tolist())
        if len(genres) > 1:
            action = "excluded_conflicting_labels"
            keep_ids.update()
        else:
            keep_ids.add(ids[0])
            action = "kept_first_removed_exact_copies" if len(ids) > 1 else "unique"
        if len(ids) > 1:
            report_rows.append({
                "sha256": digest,
                "track_ids": ids,
                "genres": genres,
                "recordings_in_group": len(ids),
                "action": action,
            })

    cleaned = merged[merged.track_id.isin(keep_ids)].drop(columns=["sha256"])
    cleaned = cleaned.sort_values("track_id").reset_index(drop=True)
    report = pd.DataFrame(
        report_rows,
        columns=["sha256", "track_ids", "genres", "recordings_in_group", "action"],
    )
    if cleaned.empty:
        raise ValueError("No recordings remain after resolving exact duplicates and conflicting labels.")
    if cleaned.track_id.duplicated().any():
        raise AssertionError("Duplicate track IDs remain after duplicate resolution.")
    return cleaned, report


def split_tracks(
    tracks: pd.DataFrame,
    *,
    test_size: float = 0.20,
    validation_size: float = 0.20,
    random_seed: int = 42,
):
    """Stratified train/validation/test split by unique original recording."""
    required = {"track_id", "genre"}
    missing = required.difference(tracks.columns)
    if missing:
        raise ValueError(f"tracks is missing required columns: {sorted(missing)}")
    if tracks.track_id.duplicated().any():
        raise ValueError("Each recording must have a unique track_id before splitting.")
    if not 0 < test_size < 1 or not 0 < validation_size < 1:
        raise ValueError("Split fractions must be between 0 and 1.")
    counts = tracks.genre.value_counts()
    if len(counts) < 2 or counts.min() < 3:
        raise ValueError("Need at least 3 recordings per genre to create stratified train/validation/test splits.")
    development, test = train_test_split(
        tracks, test_size=test_size, random_state=random_seed, stratify=tracks.genre
    )
    if development.genre.value_counts().min() < 2:
        raise ValueError("Not enough recordings per genre remain for a stratified train/validation split.")
    train, validation = train_test_split(
        development,
        test_size=validation_size,
        random_state=random_seed,
        stratify=development.genre,
    )
    return tuple(
        part.sort_values("track_id").reset_index(drop=True)
        for part in (train, validation, test)
    )
