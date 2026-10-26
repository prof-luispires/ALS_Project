"""
train_dqn_v2.py
Canonical Stable-Baselines3 DQN training configuration.

Important methodological point:
The canonical dataset contains independent channel-selection contexts rather
than a defensible temporal sequence. Accordingly, gamma=0.0 is used so the
agent optimizes the immediate channel-selection reward and does not imply
long-horizon temporal dynamics that are absent from the data.
"""
from pathlib import Path
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from hybrid_channel_env_v2 import HybridChannelEnvV2

SEEDS = [11,22,33,44,55]
TOTAL_TIMESTEPS = 100_000

DQN_PARAMS = dict(
    learning_rate=1e-3,
    buffer_size=50_000,
    learning_starts=1_000,
    batch_size=64,
    tau=1.0,
    gamma=0.0,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=500,
    exploration_fraction=0.20,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    policy_kwargs=dict(net_arch=[64,64]),
    verbose=0,
)

def train_one(df, state_mode, seed, output_dir):
    env = Monitor(HybridChannelEnvV2(
        df, state_mode=state_mode, episode_length=256, seed=seed
    ))
    model = DQN("MlpPolicy", env, seed=seed, **DQN_PARAMS)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    path = Path(output_dir)/f"dqn_{state_mode}_seed{seed}"
    model.save(path)
    return path
