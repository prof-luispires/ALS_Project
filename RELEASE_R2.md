# JWCN R2 reproducibility release candidate

**Status:** local release candidate, NOT yet published to GitHub and NOT an immutable Git commit.

## Canonical R2 implementation

- DQN entry point: `src/train_dqn_pytorch.py` (custom PyTorch contextual one-step policy learning; gamma=0).
- Environment-only state: 5 inputs. Full-information state: 11 inputs.
- DQN parameters and seeds: `results/dqn/dqn_pytorch_metadata.json`.
- Canonical data: `data/canonical/`.
- Trained DQN weights: `models/dqn/` (5 seeds x 2 state modes).
- Reported results: `results/`.
- Physical channel and reference-policy code: `src/`.

## Scope and limitations

This package contains the executable DQN training implementation and dataset-generation code. It includes preserved supervised, cross-scenario and noise-robustness outputs, **but not all executable scripts that originally generated those outputs**. Therefore, full end-to-end reproduction of every reported experiment is not yet established.

The environment requirements are unpinned in `requirements.txt`; see `ENVIRONMENT_R2.md` for versions reported for R2. A fresh reproduction run has not been performed as part of packaging.

## Publishing instructions

1. Publish these files under a clearly named `r2-reproducibility/` directory or a clean dedicated R2 branch.
2. Archive older Stable-Baselines3 scripts in `legacy/` or mark them prominently as **NOT USED FOR R2**; do not silently delete historical code.
3. Add the missing supervised/cross-scenario/noise experiment scripts if available, or keep the scope limitation explicit.
4. Commit, tag (e.g. `jwcn-r2-v1.0`), and create a GitHub Release. Record the immutable commit SHA.
5. Replace the placeholder in the manuscript Reference [20] and reviewer response with the actual commit or release URL.
6. Verify that public GitHub contents and hashes match this release and reproduce key reported results on a fresh environment.

The SHA256SUMS.txt file in this package is regenerated for this release candidate.
