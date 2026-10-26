"""
hybrid_channel_env_v2.py
Canonical DQN environment for the Springer major revision.

The environment uses rows from the canonical physics-generated dataset.
Actions: 0=RF, 1=FSO, 2=THz.
State modes:
  - environmental: distance, rain, humidity, visibility, Cn2
  - full: environmental + SNR and Eb for RF/FSO/THz

Reward is aligned with the canonical reference policy rather than max-SNR:
  +1.0  if action equals the canonical selected channel
  plus a bounded energy-efficiency term for feasible actions;
  -1.0  for selecting an infeasible channel when at least one feasible channel exists.
For all-channel-infeasible rows, RF is the explicit fallback reference.

Each step samples an independent canonical row. Therefore this is a contextual
(one-step) channel-selection MDP rather than a temporal propagation process.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

ENV_FEATURES = ["distance_m","rain_rate_mmph","humidity_pct","visibility_km","Cn2"]
FULL_FEATURES = ENV_FEATURES + [
    "SNR_RF_dB","SNR_FSO_dB","SNR_THz_dB",
    "Eb_RF_J_per_bit","Eb_FSO_J_per_bit","Eb_THz_J_per_bit"
]

class HybridChannelEnvV2(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, dataframe: pd.DataFrame, state_mode="full",
                 episode_length=256, seed=None):
        super().__init__()
        self.df = dataframe.reset_index(drop=True).copy()
        self.state_mode = state_mode
        self.features = ENV_FEATURES if state_mode == "environmental" else FULL_FEATURES
        self.episode_length = int(episode_length)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.features),), dtype=np.float32
        )
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.idx = 0

    def _sample_index(self):
        return int(self.rng.integers(0, len(self.df)))

    def _obs(self):
        return self.df.loc[self.idx, self.features].to_numpy(dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.idx = self._sample_index()
        return self._obs(), {}

    def step(self, action):
        row = self.df.iloc[self.idx]
        action = int(action)
        reference = int(row["best_channel"])
        feasible = np.array(
            [row["feasible_RF"], row["feasible_FSO"], row["feasible_THz"]],
            dtype=bool
        )
        eb = np.array(
            [row["Eb_RF_J_per_bit"], row["Eb_FSO_J_per_bit"], row["Eb_THz_J_per_bit"]],
            dtype=float
        )

        if feasible.any():
            if not feasible[action]:
                reward = -1.0
            else:
                feasible_eb = eb[feasible]
                e_min = max(float(np.min(feasible_eb)), 1e-30)
                # [0,1], equal to 1 for the minimum-Eb feasible channel.
                energy_score = float(np.clip(e_min / max(eb[action], 1e-30), 0.0, 1.0))
                reward = 0.5 * energy_score
                if action == reference:
                    reward += 1.0
        else:
            # Canonical explicit RF fallback under total infeasibility.
            reward = 1.0 if action == reference else -0.5

        self.t += 1
        terminated = False
        truncated = self.t >= self.episode_length

        info = {
            "reference_channel": reference,
            "action_correct": int(action == reference),
            "all_channels_infeasible": int(row["all_channels_infeasible"]),
            "scenario": row["scenario"],
            "realization_id": int(row["realization_id"]),
        }

        self.idx = self._sample_index()
        return self._obs(), float(reward), terminated, truncated, info
