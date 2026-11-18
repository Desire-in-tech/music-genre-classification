import pandas as pd
import pytest

from music_genre_classification.data import split_tracks


def _tracks(per_class=10):
    return pd.DataFrame([
        {"track_id": f"{genre}_{i}", "genre": genre}
        for genre in ("blues", "jazz") for i in range(per_class)
    ])


def test_split_tracks_is_disjoint_and_preserves_all_recordings():
    train, validation, test = split_tracks(_tracks(), random_seed=7)
    ids = [set(part.track_id) for part in (train, validation, test)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert set.union(*ids) == set(_tracks().track_id)
    assert all(set(part.genre) == {"blues", "jazz"} for part in (train, validation, test))


def test_split_tracks_rejects_duplicate_recordings():
    tracks = _tracks()
    tracks.loc[1, "track_id"] = tracks.loc[0, "track_id"]
    with pytest.raises(ValueError, match="unique"):
        split_tracks(tracks)


def test_split_tracks_rejects_too_few_examples_per_class():
    with pytest.raises(ValueError, match="at least 3"):
        split_tracks(_tracks(per_class=2))
