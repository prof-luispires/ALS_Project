#!/usr/bin/env python3
"""
generate_rl_dataset.py
----------------------
Gera um dataset de transições (s, a, r, s', done) a partir do ambiente
HybridChannelEnv, para utilização em Reinforcement Learning (offline RL,
análise estatística, etc.).

Saídas:
    - dataset_rl_transitions.csv
    - dataset_rl_transitions.xlsx

Cada linha corresponde a um passo:
    - episode
    - t (passo no episódio)
    - obs_* (estado normalizado antes da ação)
    - next_obs_* (estado normalizado seguinte)
    - distance_m, rain_rate_mmph, humidity_pct, visibility_km, Cn2
    - action, reward, done
    - best_channel, snr_*_db, Eb_*_J, outage flags
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hybrid_channel_env import HybridChannelEnv


def generate_rl_dataset(
    n_episodes: int = 200,
    max_steps: int = 50,
    seed: int | None = 123,
) -> pd.DataFrame:
    env = HybridChannelEnv(max_steps=max_steps, seed=seed)
    transitions = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        t = 0

        while not done:
            # Política simples: escolha aleatória (baseline)
            action = env.action_space.sample()

            next_obs, reward, terminated, truncated, info_step = env.step(action)
            done = terminated or truncated

            env_vars = info_step["env_vars"]
            metrics = info_step["metrics"]

            row = {
                "episode": ep,
                "t": t,
                "obs_d_norm": float(obs[0]),
                "obs_rain_norm": float(obs[1]),
                "obs_hum_norm": float(obs[2]),
                "obs_vis_norm": float(obs[3]),
                "next_obs_d_norm": float(next_obs[0]),
                "next_obs_rain_norm": float(next_obs[1]),
                "next_obs_hum_norm": float(next_obs[2]),
                "next_obs_vis_norm": float(next_obs[3]),
                "distance_m": env_vars["distance_m"],
                "rain_rate_mmph": env_vars["rain_rate_mmph"],
                "humidity_pct": env_vars["humidity_pct"],
                "visibility_km": env_vars["visibility_km"],
                "Cn2": env_vars["Cn2"],
                "action": int(info_step["action"]),
                "reward": float(info_step["reward"]),
                "done": bool(done),
                "best_channel": int(info_step["best_channel"]),
                "snr_rf_db": metrics["snr_rf_db"],
                "snr_fso_db": metrics["snr_fso_db"],
                "snr_thz_db": metrics["snr_thz_db"],
                "Eb_RF_J": metrics["Eb_RF_J"],
                "Eb_FSO_J": metrics["Eb_FSO_J"],
                "Eb_THz_J": metrics["Eb_THz_J"],
                "outage_RF": metrics["outage_RF"],
                "outage_FSO": metrics["outage_FSO"],
                "outage_THz": metrics["outage_THz"],
            }
            transitions.append(row)

            obs = next_obs
            t += 1

    df = pd.DataFrame(transitions)
    return df


def main():
    df = generate_rl_dataset(n_episodes=200, max_steps=50, seed=123)
    df.to_csv("dataset_rl_transitions.csv", index=False)
    df.to_excel("dataset_rl_transitions.xlsx", index=False)
    print("Dataset RL gerado:")
    print(" - dataset_rl_transitions.csv")
    print(" - dataset_rl_transitions.xlsx")
    print("Nº de transições:", len(df))


if __name__ == "__main__":
    main()
