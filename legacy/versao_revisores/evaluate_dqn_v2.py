"""
evaluate_dqn_v2.py
Evaluation definition for canonical DQN experiments.

Compare:
1. analytical/reference policy (upper-reference agreement = 1 by definition),
2. trained DQN,
3. random action baseline,
4. most-frequent-class baseline.

Report accuracy/agreement, macro-F1, infeasible-action rate, mean reward,
and results separately for feasible and all-channel-infeasible samples.
Evaluation must use held-out complete realizations, never training rows.
"""
