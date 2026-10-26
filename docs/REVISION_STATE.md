# Major-revision experimental state

## Dataset
16,000 rows: 4 scenarios x 10 atmospheric realizations x 400 distances.

Best-channel counts (RF / FSO / THz):
- Clear: 0 / 4000 / 0
- Rain: 473 / 3527 / 0
- Fog: 1231 / 2736 / 33
- Worst: 2497 / 710 / 793

All-channel-infeasible rates:
- Clear: 0.00%
- Rain: 9.68%
- Fog: 18.50%
- Worst: 25.38%

## Matched supervised results
Environmental-only RF: Accuracy 0.9684 +/- 0.0126; Macro-F1 0.9550 +/- 0.0181.
Full-information RF: Accuracy 0.9978 +/- 0.0016; Macro-F1 0.9933 +/- 0.0048.

## Strict unseen Worst scenario
Train Clear+Rain+Fog; test Worst.
Environmental-only RF: Accuracy 0.8169 +/- 0.0006; Macro-F1 0.6086 +/- 0.0006.
Full-information RF: Accuracy 0.9475 +/- 0.0202; Macro-F1 0.9141 +/- 0.0294.

## SNR-noise robustness
Full-information RF Accuracy / Macro-F1:
- 0 dB: 0.9978 / 0.9933
- 1 dB: 0.9885 / 0.9680
- 2 dB: 0.9760 / 0.9366
- 3 dB: 0.9655 / 0.9098
- 5 dB: 0.9472 / 0.8675

Environmental-only RF is unaffected by SNR measurement noise because SNR/Eb are not inputs.

## PyTorch DQN
Environmental-only: Accuracy 0.955125 +/- 0.015946; Macro-F1 0.920140 +/- 0.034764; infeasible-action rate 1.9875%.
Full-information: Accuracy 0.996625 +/- 0.001630; Macro-F1 0.993549 +/- 0.002466; infeasible-action rate 0.0562%.

Baselines: analytical reference = 1.0 by construction; most-frequent-class Accuracy 0.696875; random Accuracy 0.324813.

Interpretation: DQN is viable as a reward-oriented policy learner, but no superiority over supervised ML is claimed in the current contextual one-step setting.
