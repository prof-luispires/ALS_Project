"""
evaluate_noise_robustness.py
Canonical SNR-measurement-noise experiment for the Springer major revision.

Noise model:
    SNR_obs = SNR_true + N(0, sigma^2), sigma in {0,1,2,3,5} dB

For full-information features, capacity and transmit energy/bit are recomputed
from the noisy SNR observations, preventing leakage from clean Eb values.
Labels remain those defined by the clean physical reference policy.

Evaluation:
- grouped realization-level holdout,
- 2/10 realizations per scenario held out,
- 10 seeds,
- models trained on clean data,
- Accuracy, Macro-F1, Macro-Precision, Macro-Recall.
"""
# The executable implementation used for the reported results is preserved
# in noise_robustness_per_run.csv + metadata. This file documents the canonical
# experimental definition for the revision repository.
