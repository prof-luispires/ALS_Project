"""
reference_policy.py
-------------------
Canonical deterministic reference policy for channel labeling/evaluation.

The same rule must be used consistently by:
- dataset generation,
- supervised-learning target labels,
- analytical baseline evaluation,
- DQN reference-policy comparisons.

Rule:
1) A channel is feasible if:
       SNR >= MIN_SNR_DB
   and Capacity >= MIN_CAPACITY_BPS.
2) Among feasible channels, choose minimum energy per bit.
3) If no channel is feasible, fall back to RF (channel 0).

This corresponds to the decision rule implemented by the current
generate_scenario_dataset.py and avoids the contradictory "argmax SNR"
definition found in older DQN scripts.
"""

from __future__ import annotations

import numpy as np

from config import MIN_CAPACITY_BPS, MIN_SNR_DB


def select_reference_channel(
    snr_db,
    capacity_bps,
    energy_per_bit_j,
    *,
    min_snr_db: float = MIN_SNR_DB,
    min_capacity_bps: float = MIN_CAPACITY_BPS,
    fallback_channel: int = 0,
):
    """
    Vectorized reference-channel selection.

    Parameters
    ----------
    snr_db : array-like, shape (3, N) or (3,)
    capacity_bps : array-like, shape (3, N) or (3,)
    energy_per_bit_j : array-like, shape (3, N) or (3,)
    min_snr_db : minimum acceptable SNR
    min_capacity_bps : minimum acceptable capacity
    fallback_channel : channel used when all three links are infeasible

    Returns
    -------
    best_channel : ndarray or int
        0=RF, 1=FSO, 2=THz
    """
    snr = np.asarray(snr_db, dtype=float)
    cap = np.asarray(capacity_bps, dtype=float)
    eb = np.asarray(energy_per_bit_j, dtype=float)

    if snr.shape != cap.shape or snr.shape != eb.shape:
        raise ValueError("snr_db, capacity_bps, and energy_per_bit_j must have identical shapes.")

    if snr.ndim == 1:
        if snr.shape[0] != 3:
            raise ValueError("1D inputs must contain exactly 3 channels.")
        valid = (snr >= min_snr_db) & (cap >= min_capacity_bps)
        if not np.any(valid):
            return int(fallback_channel)
        penalized = np.where(valid, eb, np.inf)
        return int(np.argmin(penalized))

    if snr.ndim != 2 or snr.shape[0] != 3:
        raise ValueError("2D inputs must have shape (3, N).")

    valid = (snr >= min_snr_db) & (cap >= min_capacity_bps)
    penalized = np.where(valid, eb, np.inf)

    best = np.argmin(penalized, axis=0).astype(int)
    all_invalid = ~valid.any(axis=0)
    best[all_invalid] = int(fallback_channel)

    return best


def feasibility_mask(
    snr_db,
    capacity_bps,
    *,
    min_snr_db: float = MIN_SNR_DB,
    min_capacity_bps: float = MIN_CAPACITY_BPS,
):
    """Return the Boolean feasibility mask used by the reference policy."""
    snr = np.asarray(snr_db, dtype=float)
    cap = np.asarray(capacity_bps, dtype=float)
    return (snr >= min_snr_db) & (cap >= min_capacity_bps)
