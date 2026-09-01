"""Group-aware data splitting for external volume validation.

Implements the split policy from the V2 dataset plan
(``03_dataset_and_annotation_plan.md`` §5–6):

* 70 % train / 15 % validation / 15 % test.
* **Group-aware (stratified)**: each source group (L01, L02, T01) is
  proportionally represented in every split. This prevents leakage and
  ensures that the test set contains samples from all three fragmentation
  conditions.
* No sample from the train split may appear in the test split.

Leakage controls (§6 of the dataset plan):
1. External data validate **only** the ``2.5D → volume`` module.
2. External test objects are kept out of threshold / model selection.
3. Split is by group with 70/15/15 ratio.
4. Main-scene validation and external volume validation are separate
   evidence streams.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DataSplit:
    """Indices for train / validation / test splits."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray

    @property
    def n_train(self) -> int:
        return len(self.train_idx)

    @property
    def n_val(self) -> int:
        return len(self.val_idx)

    @property
    def n_test(self) -> int:
        return len(self.test_idx)

    def to_dict(self) -> dict:
        return {
            "train_idx": self.train_idx.tolist(),
            "val_idx": self.val_idx.tolist(),
            "test_idx": self.test_idx.tolist(),
        }


def group_aware_split(
    groups: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> DataSplit:
    """Stratified split that preserves group proportions in each subset.

    Parameters
    ----------
    groups : (n,) array of group labels (e.g. "L01", "L02", "T01").
    train_ratio, val_ratio, test_ratio : must sum to 1.0.
    seed : random seed for reproducibility.

    Returns
    -------
    DataSplit with train_idx, val_idx, test_idx.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Ratios must sum to 1.0"
    )

    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)

    train_idx, val_idx, test_idx = [], [], []

    for g in unique_groups:
        g_mask = groups == g
        g_indices = np.where(g_mask)[0]
        n_g = len(g_indices)

        # Shuffle within group
        g_indices = g_indices.copy()
        rng.shuffle(g_indices)

        n_train = int(round(n_g * train_ratio))
        n_val = int(round(n_g * val_ratio))
        n_test = n_g - n_train - n_val

        # Ensure at least 1 in test if group has ≥ 3 samples
        if n_g >= 3 and n_test < 1:
            n_test = 1
            n_val = max(0, n_g - n_train - n_test)

        train_idx.extend(g_indices[:n_train].tolist())
        val_idx.extend(g_indices[n_train:n_train + n_val].tolist())
        test_idx.extend(g_indices[n_train + n_val:].tolist())

    # Final shuffle to mix groups
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    # Verify no overlap
    all_train = set(train_idx.tolist())
    all_val = set(val_idx.tolist())
    all_test = set(test_idx.tolist())
    assert len(all_train & all_val) == 0, "Train/val overlap detected!"
    assert len(all_train & all_test) == 0, "Train/test overlap detected!"
    assert len(all_val & all_test) == 0, "Val/test overlap detected!"

    # Log group distribution
    for g in unique_groups:
        g_mask = groups == g
        g_total = g_mask.sum()
        g_train = np.isin(train_idx, np.where(g_mask)[0]).sum()
        g_val = np.isin(val_idx, np.where(g_mask)[0]).sum()
        g_test = np.isin(test_idx, np.where(g_mask)[0]).sum()
        logger.info(
            "  Group %s: total=%d, train=%d, val=%d, test=%d",
            g, g_total, g_train, g_val, g_test,
        )

    return DataSplit(train_idx, val_idx, test_idx)
