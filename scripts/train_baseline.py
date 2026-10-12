"""Train the baseline bundled model and save it to models/weld_rf.joblib.

Run once at repo-init time; the Docker image bakes this artifact in.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from weld_ml_service.model import save_bundle, train_bundle

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

if __name__ == "__main__":
    bundle = train_bundle()
    save_bundle(bundle)
    print(f"saved model with 5-fold CV accuracy = {bundle.accuracy_cv:.4f}")
