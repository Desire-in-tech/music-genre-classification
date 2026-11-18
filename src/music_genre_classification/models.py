"""Scikit-learn baselines for tabular audio descriptors."""
from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_baseline_models(random_seed: int = 42, n_jobs: int = -1) -> dict[str, Pipeline]:
    """Create reproducible baselines; scaling is fitted inside each training pipeline."""
    common = [("imputer", SimpleImputer(strategy="median"))]
    return {
        "svm_rbf": Pipeline(common + [
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", class_weight="balanced")),
        ]),
        "random_forest": Pipeline(common + [
            ("classifier", RandomForestClassifier(
                n_estimators=400, class_weight="balanced", random_state=random_seed,
                n_jobs=n_jobs,
            )),
        ]),
        "extra_trees": Pipeline(common + [
            ("classifier", ExtraTreesClassifier(
                n_estimators=400, class_weight="balanced", random_state=random_seed,
                n_jobs=n_jobs,
            )),
        ]),
    }
