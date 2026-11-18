"""Dataset discovery, auditing and leakage-safe recording-level splits."""
from __future__ import annotations
import hashlib, os
from pathlib import Path
import pandas as pd
import soundfile as sf
from sklearn.model_selection import train_test_split

AUDIO_EXTENSIONS={".wav",".mp3",".flac",".ogg",".au"}

def _find_audio_root(root: Path) -> Path:
    if root.name == "genres_original" and root.is_dir(): return root
    candidates=[root/"genres_original", root/"Data"/"genres_original", root/"GTZAN"/"genres_original"]
    candidates += sorted(root.glob("**/genres_original"))
    for candidate in candidates:
        if candidate.is_dir() and any(p.is_dir() for p in candidate.iterdir()): return candidate
    raise FileNotFoundError(f"Could not locate a genres_original directory under {root}. Set MUSIC_DATA_DIR to the extracted GTZAN folder.")

def discover_tracks(dataset_root: str|Path|None=None) -> pd.DataFrame:
    root=Path(dataset_root or os.getenv("MUSIC_DATA_DIR","data")).expanduser()
    audio_root=_find_audio_root(root)
    rows=[]
    for genre_dir in sorted(p for p in audio_root.iterdir() if p.is_dir()):
        for path in sorted(p for p in genre_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS):
            rows.append({"track_id":f"{genre_dir.name}/{path.name}","path":str(path),"genre":genre_dir.name})
    if not rows: raise ValueError(f"No supported audio files found in {audio_root}")
    frame=pd.DataFrame(rows).sort_values("track_id").reset_index(drop=True)
    if frame.track_id.duplicated().any(): raise ValueError("Duplicate track identifiers detected.")
    return frame

def sha256_file(path: str|Path, chunk_size: int=1024*1024) -> str:
    digest=hashlib.sha256()
    with open(path,"rb") as stream:
        for chunk in iter(lambda:stream.read(chunk_size),b""): digest.update(chunk)
    return digest.hexdigest()

def audit_tracks(tracks: pd.DataFrame, hash_files: bool=True) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Return per-file metadata and exact-content duplicate groups."""
    rows=[]
    for record in tracks.itertuples(index=False):
        row={"track_id":record.track_id,"genre":record.genre,"path":record.path}
        try:
            info=sf.info(record.path)
            row.update(duration_seconds=float(info.duration),sample_rate=int(info.samplerate),channels=int(info.channels),frames=int(info.frames),readable=True)
            if hash_files: row["sha256"]=sha256_file(record.path)
        except Exception as exc:
            row.update(readable=False,error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    audit=pd.DataFrame(rows)
    duplicates=(audit[audit.sha256.duplicated(keep=False)].sort_values("sha256")
                if hash_files and "sha256" in audit else pd.DataFrame())
    return audit,duplicates

def split_tracks(tracks: pd.DataFrame, *, test_size: float=.20, validation_size: float=.20, random_seed: int=42):
    """Stratified train/validation/test split by original recording; validation_size is fraction of development data."""
    required={"track_id","genre"}
    missing=required.difference(tracks.columns)
    if missing: raise ValueError(f"tracks is missing required columns: {sorted(missing)}")
    if tracks.track_id.duplicated().any(): raise ValueError("Each recording must have a unique track_id before splitting.")
    if not 0<test_size<1 or not 0<validation_size<1: raise ValueError("Split fractions must be between 0 and 1.")
    counts=tracks.genre.value_counts()
    if len(counts)<2 or counts.min()<3: raise ValueError("Need at least 3 recordings per genre to create stratified train/validation/test splits.")
    development,test=train_test_split(tracks,test_size=test_size,random_state=random_seed,stratify=tracks.genre)
    if development.genre.value_counts().min()<2: raise ValueError("Not enough recordings per genre remain for a stratified train/validation split.")
    train,validation=train_test_split(development,test_size=validation_size,random_state=random_seed,stratify=development.genre)
    return tuple(part.sort_values("track_id").reset_index(drop=True) for part in (train,validation,test))
