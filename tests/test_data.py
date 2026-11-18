import pandas as pd
import pytest

from music_genre_classification.data import prepare_modeling_tracks, split_tracks


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


def test_prepare_modeling_tracks_deduplicates_and_excludes_conflicting_labels():
    tracks = pd.DataFrame([
        {"track_id": "blues/a.au", "genre": "blues", "path": "/a"},
        {"track_id": "blues/b.au", "genre": "blues", "path": "/b"},
        {"track_id": "jazz/c.au", "genre": "jazz", "path": "/c"},
        {"track_id": "metal/d.au", "genre": "metal", "path": "/d"},
        {"track_id": "rock/e.au", "genre": "rock", "path": "/e"},
    ])
    audit = pd.DataFrame([
        {"track_id": "blues/a.au", "sha256": "same_blues"},
        {"track_id": "blues/b.au", "sha256": "same_blues"},
        {"track_id": "jazz/c.au", "sha256": "unique_jazz"},
        {"track_id": "metal/d.au", "sha256": "same_conflict"},
        {"track_id": "rock/e.au", "sha256": "same_conflict"},
    ])
    cleaned, report = prepare_modeling_tracks(tracks, audit)
    assert set(cleaned.track_id) == {"blues/a.au", "jazz/c.au"}
    assert len(report) == 2
    actions = dict(zip(report.sha256, report.action))
    assert actions["same_blues"] == "kept_first_removed_exact_copies"
    assert actions["same_conflict"] == "excluded_conflicting_labels"
