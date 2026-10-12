"""Baseline sklearn RandomForest for laser-weld quality classification.

Self-contained: trains on synthetic weld features at module build time so the
service has no external data dependency. In a real deployment this would be
swapped for the actual trained model, loaded from an object store.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "weld_rf.joblib"

CLASS_NAMES = ["good", "spatter", "porosity", "undercut", "cold_weld"]
FEATURE_NAMES = [
    "peak_power_W",
    "pulse_duration_ms",
    "spot_diameter_mm",
    "travel_speed_mm_s",
    "assist_gas_flow_lpm",
]


def generate_synthetic_training_data(n_samples: int = 5000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate physics-inspired synthetic weld features and quality labels.

    The decision surface mirrors reality enough for a demo: each feature has
    an acceptable range, and defects become more probable as features drift
    from their sweet spot. Not meant to be scientifically accurate.
    """
    rng = np.random.default_rng(seed)
    X = np.column_stack([
        rng.uniform(300.0, 4000.0, n_samples),   # peak power W
        rng.uniform(0.5, 20.0, n_samples),       # pulse duration ms
        rng.uniform(0.2, 1.2, n_samples),        # spot diameter mm
        rng.uniform(1.0, 50.0, n_samples),       # travel speed mm/s
        rng.uniform(5.0, 25.0, n_samples),       # assist gas flow lpm
    ])

    # Normalised distance from each sweet spot
    sweet_spots = np.array([
        (300 + 4000) / 2,   # peak power
        (0.5 + 20) / 2,     # pulse duration
        (0.2 + 1.2) / 2,    # spot diameter
        (1 + 50) / 2,       # travel speed
        (5 + 25) / 2,       # gas flow
    ])
    ranges = np.array([3700, 19.5, 1.0, 49.0, 20.0])
    normalised = np.abs(X - sweet_spots) / ranges

    # 60% good welds, 40% split across 4 defect modes correlated with specific drifts
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        total_drift = normalised[i].sum()
        if total_drift < 1.0 and rng.random() < 0.9:
            y[i] = 0   # good
            continue
        # Each feature extreme biases toward a specific defect
        drifts = normalised[i]
        if drifts[0] > drifts.mean() and rng.random() < 0.7:
            y[i] = 1   # spatter — power too high/low
        elif drifts[4] > drifts.mean() and rng.random() < 0.7:
            y[i] = 2   # porosity — gas flow off
        elif drifts[3] > drifts.mean() and rng.random() < 0.7:
            y[i] = 3   # undercut — travel speed off
        else:
            y[i] = 4   # cold weld — generic drift
    return X, y


@dataclass
class ModelBundle:
    estimator: RandomForestClassifier
    classes: list[str]
    features: list[str]
    version: str
    accuracy_cv: float   # recorded at training time


def train_bundle(seed: int = 42) -> ModelBundle:
    """Train the baseline RF on synthetic data, return a bundle."""
    from sklearn.model_selection import cross_val_score
    X, y = generate_synthetic_training_data(seed=seed)
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=6, min_samples_leaf=5,
        random_state=seed, n_jobs=1,
    )
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring="accuracy")
    rf.fit(X, y)
    logger.info("trained RF — 5-fold CV accuracy = %.3f ± %.3f",
                cv_scores.mean(), cv_scores.std())
    return ModelBundle(
        estimator=rf,
        classes=CLASS_NAMES,
        features=FEATURE_NAMES,
        version="0.1.0",
        accuracy_cv=float(cv_scores.mean()),
    )


def save_bundle(bundle: ModelBundle, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "estimator": bundle.estimator,
        "classes": bundle.classes,
        "features": bundle.features,
        "version": bundle.version,
        "accuracy_cv": bundle.accuracy_cv,
    }, path)
    logger.info("saved bundle -> %s", path)


def load_bundle(path: Path = MODEL_PATH) -> ModelBundle:
    data = joblib.load(path)
    return ModelBundle(**data)
