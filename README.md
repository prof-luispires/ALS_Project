# Hybrid RF/FSO/THz — JWCN R2 reproducibility release candidate

**Authoritative R2 implementation:** `src/train_dqn_pytorch.py`, not the older Stable-Baselines3 scripts. This is a local release candidate; GitHub publication and full independent reproduction have **not** been verified.

## Repository layout

- `src/`: R2 physical models, dataset generation, reference policy and **PyTorch contextual one-step DQN**.
- `data/canonical/`: preserved 16,000-row synthetic dataset and scenario metadata.
- `models/dqn/`: saved DQN weights for five seeds and two state representations.
- `results/`: preserved DQN, supervised, cross-scenario and noise robustness outputs.
- `legacy/evolucao_2/`: earlier experiment scripts, including Stable-Baselines3 and supervised scripts; **NOT AUTHORITATIVE FOR R2**. Do not use these to claim reproduction of the R2 tables.
- `legacy/versao_revisores/`: historical review-stage DQN variants; **NOT AUTHORITATIVE FOR R2**.
- `docs/`: provenance and verification notes.

## Key R2 DQN settings

- Framework: custom PyTorch implementation; contextual one-step DQN, gamma = 0.
- Input: 5 environmental-only variables or 11 full-information variables (5 environmental + 3 SNR + 3 energy-per-bit).
- Replay buffer: 50,000; 100,000 interactions; batch size 64; exploration linear over first 20%; no target network.
- Random seeds: 11, 22, 33, 44, 55.
- See `results/dqn/dqn_pytorch_metadata.json` for preserved run metadata.

## Running and provenance

Create a Python 3.10.11 environment; install `requirements-r2.txt` (PyTorch CPU wheel `2.9.1+cpu` as reported; the exact wheel build requires an appropriate PyTorch package index). Inspect `src/EXPERIMENTS.md`, the script CLI, and the preserved dataset metadata before executing. Do not assume that historical supervised scripts are compatible with the canonical R2 dataset or its realization-level evaluation protocols.

`SHA256SUMS.txt` lists the package file hashes. After publication, create a GitHub tag/release and record the **full commit SHA** in manuscript Reference [20] and the reviewer response. GitHub commit SHA is not supplied by this ZIP.

## Reproducibility limitations

The provided source does not establish an independently rerun, end-to-end reproduction of every supervised, cross-scenario and noise experiment. Their result artifacts are preserved, and candidate historical scripts are segregated under `legacy/`. A fresh DQN training run has not been executed during packaging. See `docs/PROVENANCE_AUDIT_R2.md`.
