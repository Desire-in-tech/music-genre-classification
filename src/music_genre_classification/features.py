"""Track-level audio representations and cache helpers."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

def extract_track_features(audio_path, *, sample_rate=22050, n_mfcc=20, duration=30):
    import librosa
    y,sr=librosa.load(str(audio_path),sr=sample_rate,mono=True,duration=duration)
    if y.size==0: raise ValueError(f"Empty or unreadable audio: {audio_path}")
    descriptors={
      "mfcc":librosa.feature.mfcc(y=y,sr=sr,n_mfcc=n_mfcc),
      "chroma":librosa.feature.chroma_stft(y=y,sr=sr),
      "spectral_centroid":librosa.feature.spectral_centroid(y=y,sr=sr),
      "spectral_bandwidth":librosa.feature.spectral_bandwidth(y=y,sr=sr),
      "spectral_rolloff":librosa.feature.spectral_rolloff(y=y,sr=sr),
      "zero_crossing_rate":librosa.feature.zero_crossing_rate(y),
      "rms":librosa.feature.rms(y=y),
    }
    result={}
    for name,values in descriptors.items():
        arr=np.asarray(values,dtype=np.float64)
        if arr.ndim==1: arr=arr[None,:]
        for i,row in enumerate(arr):
            prefix=f"{name}_{i}" if len(arr)>1 else name
            result[f"{prefix}_mean"]=float(row.mean()); result[f"{prefix}_std"]=float(row.std())
    tempo,_=librosa.beat.beat_track(y=y,sr=sr)
    result["tempo"]=float(np.asarray(tempo).reshape(-1)[0])
    if not np.isfinite(list(result.values())).all(): raise ValueError(f"Non-finite feature from {audio_path}")
    return result

def build_feature_table(tracks, *, sample_rate=22050, n_mfcc=20, duration=30):
    rows=[]
    for record in tracks.itertuples(index=False):
        rows.append({"track_id":record.track_id,"genre":record.genre,**extract_track_features(record.path,sample_rate=sample_rate,n_mfcc=n_mfcc,duration=duration)})
    return pd.DataFrame(rows)

def save_feature_table(features,path):
    destination=Path(path); destination.parent.mkdir(parents=True,exist_ok=True); features.to_parquet(destination,index=False)

def extract_log_mel(audio_path, *, sample_rate=22050, n_mels=128, n_fft=2048, hop_length=1024, duration=30, target_frames=256):
    """Return a fixed 128x256 (by default) normalized log-Mel image for one whole track."""
    import librosa
    y,_=librosa.load(str(audio_path),sr=sample_rate,mono=True,duration=duration)
    if y.size==0: raise ValueError(f"Empty or unreadable audio: {audio_path}")
    mel=librosa.feature.melspectrogram(y=y,sr=sample_rate,n_fft=n_fft,hop_length=hop_length,n_mels=n_mels,power=2.0)
    logmel=librosa.power_to_db(mel,ref=np.max)
    positions=np.linspace(0,logmel.shape[1]-1,target_frames)
    resized=np.vstack([np.interp(positions,np.arange(logmel.shape[1]),row) for row in logmel])
    resized=np.clip(resized,-80,0).astype("float32")/80.0
    return resized[...,None]

def build_log_mel_dataset(tracks, *, sample_rate=22050,n_mels=128,hop_length=1024,duration=30,target_frames=256):
    images=np.stack([extract_log_mel(p,sample_rate=sample_rate,n_mels=n_mels,hop_length=hop_length,duration=duration,target_frames=target_frames) for p in tracks.path])
    return images
