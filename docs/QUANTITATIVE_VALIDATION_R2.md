# JWCN R2 — Quantitative reproducibility validation addendum

**Scope:** Audit of the local JWCN R2 package against `JWCN-D-26-00181_Revised_Manuscript_clean_REV(2).docx`. This document records actual checks, not a claim of end-to-end experimental reproduction.

## Confirmed checks

1. **Fresh physical-dataset regeneration:** Executed `src/generate_dataset.py` using the supplied default configuration in a separate temporary directory. Regenerated dataset: **16,000 rows × 25 columns**. `pandas.DataFrame.equals` returned **True** against `data/canonical/dataset_all_scenarios.csv`; every field and row matched exactly.
2. **Physical policy and infeasibility (manuscript Table 6):** Clear RF/FSO/THz = 0/4000/0; Rain = 473/3527/0; Fog = 1231/2736/33; Worst = 2497/710/793. All-channel infeasible = 0, 409, 745, 1165 respectively; total 2319/16000 (14.49%). All agree with the manuscript.
3. **Preserved output versus manuscript:** Compared **68 published mean metric cells** (accuracy, macro-F1, and DQN infeasible-action percentages) from the manuscript's Tables 12, 13, 15 and 16 against the corresponding saved summary CSVs. **68/68 matched at the precision reported in the manuscript.** Table 6 counts were independently checked separately.
4. **Per-run aggregation consistency:** For DQN, supervised, cross-scenario, and noise-robustness CSVs, recomputed grouped means of accuracy and macro-F1 from per-run records. They agree with the respective saved summary CSVs to floating-point precision (maximum difference <= 2.3e-16). Source row counts: DQN 40; supervised 60; cross-scenario 46; noise robustness 200.
5. **DQN settings documented in source:** `src/train_dqn_pytorch.py` defines the two state modes with 5 and 11 inputs, seeds 11/22/33/44/55, 100,000 interactions, buffer 50,000, batch 64, epsilon schedule over 20%, two hidden layers of 64 ReLU units, and no target network. The implementation uses an immediate contextual reward, with no discounted future-return term.

## Remaining limitations

- A **fresh full DQN training run was not performed** in this audit. Saved PyTorch model weights and per-seed outputs were not independently regenerated.
- **Supervised, cross-scenario, and noise experiments were not independently rerun**; the exact original generating scripts for these final R2 experiments are not fully established in the canonical package. Numeric agreement between saved artifacts and manuscript is not independent reproducibility.
- Historical datasets under `legacy/` differ from the canonical dataset. This is not a reason to replace the canonical data: the canonical data regenerates exactly from the R2 generator and agrees with manuscript Table 6. The historical data should remain explicitly non-authoritative.
- The validation was run with the available local Python environment, which is not established to be the exact original manuscript environment (PyTorch 2.9.1+cpu). Do not claim environment-identical reproduction.
- The existing README and provenance notes still refer to an unpublished release candidate. Update the wording once GitHub publication and the commit SHA have been verified. A GitHub tag is not a substitute for a commit-specific reference.

## Recommended publication gate

1. Add this addendum under `docs/QUANTITATIVE_VALIDATION_R2.md` in the open pull request.
2. Update the README to describe the actual version-controlled package, without asserting full independent reproduction.
3. Verify that no obsolete root-level training files remain after merging. The legacy scripts may remain under `legacy/`.
4. If the reviewer requires independently reproducible supervised outputs, supply or reconstruct the exact scripts/protocols and rerun them before making that claim.
5. Merge only after confirming the file tree and commit, then create the R2 tag/release and cite its exact commit SHA.
