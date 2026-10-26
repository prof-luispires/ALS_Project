# Experiment scripts

`generate_dataset.py` generates the canonical physical dataset.

`train_dqn_pytorch.py` is the executable canonical DQN implementation used for the revision. It reproduces the two DQN state modes and baseline evaluation.

The exact supervised, cross-scenario and noise outputs used in the revision are preserved under `results/`. Their metadata files define the corresponding protocols. The original exploratory/legacy scripts were deliberately excluded from this clean Git package because several used obsolete max-SNR labels, inconsistent link budgets, or row-level random splits.
