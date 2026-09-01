"""Volume estimation methods — ablation baselines and the correction-ratio model.

Implements the ablation from the V2 spec (``05_method_flow_audit_and_v2_spec.md``):

1. **Bounding-box**      ``V_box = L × W × H``
2. **Ellipsoid**         ``V_ellipsoid = (π/6) × L × W × H``
3. **2.5-D integration**  ``V_2.5D = Σ max(z_top − z_ground, 0) × Δ²``
4. **Linear correction**  ``V_corr = α · V_2.5D`` (single scalar, interpretable)
5. **Shape-aware**        LightGBM on ratio ``r = V_true / V_2.5D``

Key design: the shape-aware model learns the **correction ratio** r, NOT the
absolute volume. This decouples "how big is this rock" (answered by V_2.5D)
from "how much does 2.5D overestimate for this shape" (answered by r). The
ratio is dimensionless and shape-dependent, making it far more transferable
across scenes than a direct volume regression.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .shape_descriptors import ShapeDescriptors

logger = logging.getLogger(__name__)

_EPS = 1e-9


def estimate_box_volume(desc: ShapeDescriptors) -> float:
    """V_box = L × W × H. Upper bound; overestimates irregular rocks."""
    return desc.V_box


def estimate_ellipsoid_volume(desc: ShapeDescriptors) -> float:
    """V_ellipsoid = (π/6) × L × W × H. Geometric approximation."""
    return desc.V_ellipsoid


def estimate_2_5d_volume(desc: ShapeDescriptors) -> float:
    """Deterministic 2.5-D integration from the height surface."""
    return desc.V_2_5d


# ── Transferable features only (dimensionless, scale-invariant) ─────────

TRANSFERABLE_FEATURES = [
    # Footprint shape
    "C",                # circularity — how round is the footprint
    "AR",               # aspect ratio — how elongated
    "solidity",         # A / A_convex — how convex
    "compactness",      # P / sqrt(A) — boundary complexity
    "eq_diam_ratio",    # eq_diameter / L — how close to circle
    # Height distribution shape
    "H_mean_norm",      # mean height / max height — how flat-topped
    "H_std_norm",       # height std / max height — how uneven
    "H_p25_norm",       # 25th percentile / max height
    "H_p75_norm",       # 75th percentile / max height
    "H_skew_norm",      # height skewness — asymmetry of height profile
    # Volume fill ratios (strongest predictors of r = V_true/V_2.5D)
    "fill_ratio",       # V_2.5d / V_box — how full is the bbox
    "ellipsoid_ratio",  # V_2.5d / V_ellipsoid — ellipsoid approximation quality
]

TRANSFERABLE_NAMES = [
    "C", "AR", "solidity", "compactness", "eq_diam_ratio",
    "H_mean_norm", "H_std_norm", "H_p25_norm", "H_p75_norm", "H_skew_norm",
    "fill_ratio", "ellipsoid_ratio",
]


def _extract_transferable_features(desc: ShapeDescriptors) -> np.ndarray:
    """Extract only dimensionless, scale-invariant features.

    These features describe *shape*, not *size*. They are the same for a
    5 mm rock and a 5 m rock with the same geometry, so a model trained on
    them transfers across scenes.
    """
    H = desc.H if desc.H > _EPS else _EPS
    return np.array([
        desc.C,                                          # circularity [0, 1]
        desc.AR,                                         # aspect ratio ≥ 1
        desc.solidity,                                   # convexity [0, 1]
        desc.compactness,                                # boundary complexity
        desc.eq_diam_ratio,                              # roundness proxy
        desc.H_mean / H,                                 # mean/max height [0, 1]
        desc.H_std / H,                                  # height std / max
        desc.H_p25 / H,                                  # 25th pct / max
        desc.H_p75 / H,                                  # 75th pct / max
        desc.H_skew,                                     # height skewness
        desc.fill_ratio,                                 # V_2.5d / V_box
        desc.ellipsoid_ratio,                            # V_2.5d / V_ellipsoid
    ])


class LinearCorrectionModel:
    """Single-parameter linear correction: V_corrected = α · V_2.5D.

    Finds α = mean(V_true / V_2.5D) on the training set. This is the most
    transferable model: one scalar that captures the average overestimation
    of the 2.5D method. It cannot overfit and works across scenes because
    it only assumes "2.5D consistently overestimates by some factor".
    """

    def __init__(self):
        self._alpha = 1.0
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def alpha(self) -> float:
        return self._alpha

    def train(
        self,
        descriptors: list[ShapeDescriptors],
        volumes_true: np.ndarray,
        **kwargs,
    ) -> dict:
        volumes_2_5d = np.array([d.V_2_5d for d in descriptors])
        ratios = volumes_true / (volumes_2_5d + _EPS)
        ratios = np.clip(ratios, 0.01, 10.0)  # remove outliers
        self._alpha = float(np.median(ratios))
        self._is_trained = True

        train_pred = self._alpha * volumes_2_5d
        train_mae = float(np.mean(np.abs(train_pred - volumes_true)))
        train_rel = float(np.mean(np.abs(train_pred - volumes_true) / (volumes_true + _EPS)))

        info = {
            "alpha": self._alpha,
            "train_mae": train_mae,
            "train_mean_rel_error": train_rel,
            "n_train": len(descriptors),
            "model_type": "linear_correction",
        }
        logger.info("LinearCorrectionModel: alpha=%.4f, train_rel=%.2f%%",
                     self._alpha, train_rel * 100)
        return info

    def predict(self, descriptors: list[ShapeDescriptors]) -> np.ndarray:
        if not self._is_trained:
            raise RuntimeError("Model not trained.")
        v_2_5d = np.array([d.V_2_5d for d in descriptors])
        return self._alpha * v_2_5d

    def feature_importance(self) -> dict:
        return {"alpha (global correction factor)": float(self._alpha)}


class ShapeAwareModel:
    """LightGBM that learns the correction ratio r = V_true / V_2.5D.

    Instead of regressing V_true directly (which would learn the *size*
    distribution of the training scene), this model regresses the
    dimensionless ratio r. At inference:

        V_predicted = r_predicted × V_2.5D

    The ratio r depends on *shape* (flat rocks → r << 1, spherical rocks →
    r ≈ 1), not on absolute scale. This makes the model transferable to
    scenes with different rock-size distributions.

    Two feature modes:
    - ``mode="transferable"``: only 5 dimensionless features (C, AR, H_mean/H,
      H_std/H, L/W). Most transferable, least expressive.
    - ``mode="full"``: all 13 features including absolute dimensions.
      More accurate on same-scene data, less transferable.
    """

    def __init__(self, params: dict | None = None, mode: str = "transferable"):
        self.params = params or {}
        self.mode = mode
        self._model = None
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def _get_features(self, descriptors: list[ShapeDescriptors]) -> np.ndarray:
        if self.mode == "transferable":
            return np.array([_extract_transferable_features(d) for d in descriptors])
        else:
            return np.array([d.to_vector() for d in descriptors])

    @property
    def _feature_names(self) -> list[str]:
        if self.mode == "transferable":
            return TRANSFERABLE_NAMES
        else:
            return ShapeDescriptors.FEATURE_NAMES

    def train(
        self,
        descriptors: list[ShapeDescriptors],
        volumes_true: np.ndarray,
        val_descriptors: list[ShapeDescriptors] | None = None,
        val_volumes: np.ndarray | None = None,
    ) -> dict:
        """Train the LightGBM ratio model.

        Target: r = V_true / V_2.5D  (dimensionless correction ratio)
        Features: transferable shape features (default) or full features.
        """
        import lightgbm as lgb

        X_train = self._get_features(descriptors)
        v_2_5d_train = np.array([d.V_2_5d for d in descriptors])
        y_train = volumes_true / (v_2_5d_train + _EPS)
        y_train = np.clip(y_train, 0.01, 10.0)

        train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=False)

        valid_sets = [train_set]
        valid_names = ["train"]

        if val_descriptors is not None and val_volumes is not None:
            X_val = self._get_features(val_descriptors)
            v_2_5d_val = np.array([d.V_2_5d for d in val_descriptors])
            y_val = val_volumes / (v_2_5d_val + _EPS)
            y_val = np.clip(y_val, 0.01, 10.0)
            val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
            valid_sets.append(val_set)
            valid_names.append("valid")

        callbacks = [lgb.log_evaluation(50)]
        if len(valid_sets) > 1:
            callbacks.append(lgb.early_stopping(100, verbose=False))

        self._model = lgb.train(
            self.params,
            train_set,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        self._is_trained = True

        best_iter = self._model.best_iteration
        ratios_pred = self._model.predict(X_train, num_iteration=best_iter)
        train_pred = ratios_pred * v_2_5d_train
        train_mae = float(np.mean(np.abs(train_pred - volumes_true)))
        train_rel = float(np.mean(np.abs(train_pred - volumes_true) / (volumes_true + _EPS)))

        info = {
            "best_iteration": int(best_iter) if best_iter > 0 else self.params.get("n_estimators", 0),
            "train_mae": train_mae,
            "train_mean_rel_error": train_rel,
            "n_features": X_train.shape[1],
            "n_train": X_train.shape[0],
            "mode": self.mode,
            "feature_names": self._feature_names,
            "target": "correction_ratio_r",
        }
        logger.info("ShapeAwareModel (ratio mode=%s) trained: %s", self.mode, info)
        return info

    def predict(self, descriptors: list[ShapeDescriptors]) -> np.ndarray:
        """Predict volumes: V = r_predicted × V_2.5D."""
        if not self._is_trained:
            raise RuntimeError("Model is not trained. Call .train() first.")
        X = self._get_features(descriptors)
        best_iter = self._model.best_iteration
        ratios = self._model.predict(X, num_iteration=best_iter)
        v_2_5d = np.array([d.V_2_5d for d in descriptors])
        return ratios * v_2_5d

    def predict_ratio(self, descriptors: list[ShapeDescriptors]) -> np.ndarray:
        """Predict correction ratios only (without multiplying by V_2.5D)."""
        if not self._is_trained:
            raise RuntimeError("Model is not trained.")
        X = self._get_features(descriptors)
        best_iter = self._model.best_iteration
        return self._model.predict(X, num_iteration=best_iter)

    def feature_importance(self) -> dict[str, float]:
        if not self._is_trained:
            return {}
        importance = self._model.feature_importance(importance_type="split")
        return dict(zip(self._feature_names, importance.tolist()))

    def save(self, path: str | Path) -> None:
        if not self._is_trained:
            raise RuntimeError("Cannot save an untrained model.")
        self._model.save_model(str(path))
        meta = {
            "params": self.params,
            "mode": self.mode,
            "feature_names": self._feature_names,
            "target": "correction_ratio_r",
        }
        meta_path = str(path) + ".meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Model saved to %s", path)

    def load(self, path: str | Path) -> None:
        import lightgbm as lgb
        self._model = lgb.Booster(model_file=str(path))
        meta_path = str(path) + ".meta.json"
        if Path(meta_path).exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self.params = meta.get("params", {})
            self.mode = meta.get("mode", "transferable")
        self._is_trained = True
        logger.info("Model loaded from %s (mode=%s)", path, self.mode)
