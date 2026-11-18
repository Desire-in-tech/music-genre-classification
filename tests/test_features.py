import pandas as pd
import pytest

from music_genre_classification.evaluation import feature_matrix


def test_feature_matrix_excludes_metadata():
    X, y = feature_matrix(pd.DataFrame({
        "track_id": ["a", "b"], "genre": ["jazz", "rock"], "mfcc_0_mean": [1.0, 2.0]
    }))
    assert list(X.columns) == ["mfcc_0_mean"]
    assert y.tolist() == ["jazz", "rock"]


def test_feature_matrix_requires_metadata():
    with pytest.raises(ValueError, match="missing"):
        feature_matrix(pd.DataFrame({"mfcc_0_mean": [1.0]}))
