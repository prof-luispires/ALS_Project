#!/usr/bin/env python3
"""
evaluate_dqn_multiscenario.py
-----------------------------
Avalia um modelo DQN treinado no ambiente multi-cenário.
"""
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from hybrid_channel_env_multiscenario import HybridChannelEnvMulti
import os

def run_episode(env, model, scenario_name):
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0
    done = False
    logs = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        logs.append({
            "scenario": scenario_name,
            "step": steps,
            "action": int(action),
            "reward": float(reward),
            "snr_rf": float(info["snrs"][0]),
            "snr_fso": float(info["snrs"][1]),
            "snr_thz": float(info["snrs"][2]),
            "best_channel": int(info["best_channel"]),
        })

        total_reward += reward
        steps += 1

    return float(total_reward), pd.DataFrame(logs)

def evaluate_model(
    model_path: str = "logs_dqn_multiscenario/dqn_multiscenario_final.zip",
    out_csv: str = "dqn_evaluation_results.csv",
    out_xlsx: str = "dqn_evaluation_results.xlsx",
    episodes_per_scenario: int = 5,
):
    results = []
    full_logs = []

    for scen in ["clear", "rain", "fog", "worst"]:
        env = HybridChannelEnvMulti()
        model = DQN.load(model_path)

        ep_rewards = []
        scenario_logs = []

        for _ in range(episodes_per_scenario):
            r, df = run_episode(env, model, scen)
            ep_rewards.append(r)
            scenario_logs.append(df)

        results.append({
            "scenario": scen,
            "mean_reward": float(np.mean(ep_rewards)),
            "std_reward": float(np.std(ep_rewards)),
            "episodes": ep_rewards,
        })

        full_logs.append(pd.concat(scenario_logs, ignore_index=True))

    df_all = pd.concat(full_logs, ignore_index=True)
    df_all.to_csv(out_csv, index=False)

    df_summary = pd.DataFrame(results)
    df_summary["episodes_str"] = df_summary["episodes"].apply(
        lambda lst: ", ".join(f"{x:.2f}" for x in lst)
    )

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df_all.to_excel(writer, sheet_name="logs", index=False)
        df_summary.to_excel(writer, sheet_name="summary", index=False)

    print("\n=== Avaliação DQN por cenário ===")
    for r in results:
        print(f"→ {r['scenario']}: mean={r['mean_reward']:.2f}, std={r['std_reward']:.2f}")

    print(f"\nResultados completos guardados em: {out_csv}")
    print(f"Resultados completos (Excel) guardados em: {out_xlsx}")

if __name__ == "__main__":
    evaluate_model()
