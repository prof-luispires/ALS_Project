# R2 source provenance and audit

## Sources

- Primary canonical package: `JWCN_R2_reproducibility_release_candidate.zip`, derived from `NEW_VERSION-hybrid-rf-fso-thz-maritime.zip`.
- Supplemental historical scripts: user-supplied `Python Files.zip`, specifically `Evolucao_2` and `Versão_revisores` directories.

## Canonical vs historical

Only `src/train_dqn_pytorch.py` is designated the R2 DQN trainer. The Stable-Baselines3 training scripts are retained under `legacy/` for provenance, not as active entry points. Supervised scripts under `legacy/` come from earlier datasets/protocols and are not asserted to generate the exact R2 outputs.

## File identity checks

- metadata_matches_reviewer_folder: **True**
- dataset_matches_reviewer_folder: **False**

## Verification status

- Canonical files and historical scripts organized: YES.
- Python syntax compilation for included scripts: see `VERIFICATION_STATUS.json`.
- Fresh end-to-end DQN training: NOT RUN.
- Reproduction of all supervised/cross-scenario/noise outputs: NOT VERIFIED.
- Public GitHub commit/release: NOT CREATED.
- Reported PyTorch build: `2.9.1+cpu`; manuscript and response letter should agree.
- Figure 12 should show 5 or 11 inputs depending on state representation, not 10.
