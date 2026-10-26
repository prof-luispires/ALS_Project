#!/usr/bin/env python3
"""
train_dqn_agent.py
------------------
Treino de um agente Deep Q-Learning (DQN) para seleção de canal
no ambiente HybridChannelEnv (RF / FSO / THz).

Dependências:
    pip install gymnasium stable-baselines3
"""
from __future__ import annotations

import os
import numpy as np
import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

from hybrid_channel_env import HybridChannelEnv


def make_env(seed: int | None = None):
    env = HybridChannelEnv(max_steps=50, seed=seed)
    return env


def main():
    log_dir = "logs_dqn"
    os.makedirs(log_dir, exist_ok=True)

    env = make_env(seed=123)
    eval_env = make_env(seed=456)

    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10,
        min_evals=5,
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        callback_after_eval=stop_callback,
    )

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=100_000,
        learning_starts=1_000,
        batch_size=64,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1_000,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.3,
        verbose=1,
    )

    total_timesteps = 200_000
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    model_path = os.path.join(log_dir, "dqn_hybrid_channel_final")
    model.save(model_path)
    print(f"Modelo DQN guardado em: {model_path}")

    # Avaliação rápida
    episodes = 10
    rewards = []
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
        print(f"Episódio {ep}: recompensa total = {ep_reward:.2f}")

    print(f"\nRecompensa média em {episodes} episódios: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")


if __name__ == "__main__":
    main()
